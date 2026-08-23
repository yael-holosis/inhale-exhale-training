"""Read the uploaded windows: the two databases, and the samples in S3.

Every connection here is **read only** - this repo has no writer. Production's is read-only by
the server's rules as well: production's account is granted SELECT and nothing else.

Every credential - host, user and password alike - comes out of Secrets Manager, and nothing is
stored in this repo or read from the environment. Production's secret lives in the production
account, so it is read under that account's own profile while the rest of the process stays on
the data-science one.

The two sides of an environment never join in SQL. On prod they are different servers, so the
signal and window frames are merged in pandas and the code path is the same on both instances -
a SQL join would work on the algo instance and fail on the other.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yaml

LABELS = "labels"
DEVICE = "device"
# The reviewer's polarity flag on `RespirationWindow`. Named once: it reaches the catalogue
# query, the shard columns and the orientation decision.
REVIEWER_FLIPPED = "ReviewerFlipped"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "parameter" / "sources" / "default.yaml"


class ReadOnly(RuntimeError):
    """Raised if anything ever tries to write. Nothing here does; the guard is the point."""


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


def env_config(env_key: str) -> dict[str, Any]:
    environments = config()["environments"]
    if env_key not in environments:
        raise KeyError(f"unknown environment {env_key!r}; known: {sorted(environments)}")
    return environments[env_key]


def profile() -> str:
    cfg = config()["aws"]
    return os.environ.get(cfg["profile_env_var"]) or cfg["profile"]


def use_profile() -> str:
    """Point the process at the configured profile and drop any cached boto3 session.

    Deliberately does **not** log in. `AwsProfileManager.select_aws_profile()` is an interactive
    picker that prompts on stdin, which hangs a build run to no purpose - the profile is named in
    config precisely so nobody has to be asked. `sign_in` is the explicit, non-interactive
    refresh, and `ensure_session` calls it when the credentials are actually unusable.
    """
    import boto3

    name = profile()
    os.environ["AWS_PROFILE"] = name
    boto3.DEFAULT_SESSION = None
    return name


def credentials_ok() -> bool:
    """Whether **boto3** can reach AWS - which is what every query here runs on.

    Asking the CLI instead gets a yes from its own credential cache while boto3 is still locked
    out, and nothing gets fixed.
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        boto3.Session().client("sts", region_name=config()["aws"]["region"]).get_caller_identity()
        return True
    except (BotoCoreError, ClientError):
        return False


@lru_cache(maxsize=1)
def ensure_session(timeout_s: int = 120) -> bool:
    """Sign in if the process cannot reach AWS. Called once, at the top of a run.

    Blocks until somebody approves the browser tab, so it belongs in a command-line entrypoint
    and nowhere else.
    """
    import subprocess

    use_profile()
    if credentials_ok():
        return True
    name = profile()
    print(f"AWS session for {name} has expired - signing in")
    try:
        subprocess.run(["aws", "sso", "login", "--profile", name], timeout=timeout_s,
                       check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as error:
        print(f"sign in failed: {error}")
        return False
    use_profile()                    # a fresh token needs a fresh session
    return credentials_ok()


# ------------------------------------------------------------------------------- credentials

def credentials_for(env_key: str, role: str) -> dict[str, str]:
    """Host, user and password for one connection. The secret is the only source.

    No environment variable and no local file: a connection whose credential is not in Secrets
    Manager is a configuration error, not something to fall back from.
    """
    cfg = env_config(env_key)[role]
    if not cfg.get("secret_name"):
        raise RuntimeError(f"no secret_name for {env_key}.{role} in {CONFIG_PATH.name}")
    return _from_secret(cfg["secret_name"], cfg.get("secret_profile"))


def _from_secret(secret_name: str, secret_profile: str | None = None) -> dict[str, str]:
    """Host, user and password out of one Secrets Manager entry.

    Read here rather than handed to `DBManager(secret_name=...)`: that path takes the database
    from the secret's own `db_name` and **ignores the database argument**. Every secret we have
    says `edge_data`, so letting it decide points the labels connection at the wrong schema -
    which is exactly how `RespirationWindow` comes back "doesn't exist".

    `secret_profile` is for a secret in another account. A named session is used rather than
    switching the environment variable, so nothing else in the process is affected.
    """
    import boto3

    use_profile()
    session = boto3.Session(profile_name=secret_profile) if secret_profile else boto3.Session()
    client = session.client("secretsmanager", region_name=config()["aws"]["region"])
    payload = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
    return {"host": payload.get("host") or payload["endpoint"],
            "user": payload["username"], "password": payload["password"]}


@lru_cache(maxsize=8)
def engine(env_key: str, role: str):
    """A SQLAlchemy engine on one side of one environment.

    A plain engine rather than a `DBManager`: every manager in the house library autoloads
    `edge_data` tables on construction, and the labelling schema does not have them.
    """
    from sqlalchemy import create_engine

    use_profile()
    cfg = env_config(env_key)[role]
    credentials = credentials_for(env_key, role)
    return create_engine(
        f"mysql+pymysql://{credentials['user']}:{quote_plus(credentials['password'])}"
        f"@{credentials['host']}/{cfg['database']}",
        pool_recycle=1200, pool_pre_ping=True)


READS = ("SELECT", "SHOW", "DESCRIBE")


def frame(env_key: str, role: str, query: str, **binds) -> pd.DataFrame:
    """One read. Binds are substituted by the driver, never formatted into the statement.

    The guard is not defence against an attacker - it is a standing assertion that this repo
    reads and nothing else, so a writer added later fails here instead of on production.
    """
    if not query.lstrip().upper().startswith(READS):
        raise ReadOnly(f"only reads are allowed here; refused: {query.strip()[:60]}")
    return pd.read_sql(query, engine(env_key, role), params=binds or None)


# ------------------------------------------------------------------------------- the catalogue

def windows_frame(env_key: str) -> pd.DataFrame:
    """Every uploaded window with its human-span counts. Labels side only.

    `Spans` is the number of `BreathPhaseTimeRecord` rows on the window - the only thing that
    says whether a person has labelled it. There is no boolean "labelled" column, and filtering
    on one that is not there silently keeps every row.
    """
    t = config()["tables"]
    return frame(env_key, LABELS, f"""
        SELECT w.ID, w.RadarSignalID, w.WindowIndex, w.WaveformS3Path,
               w.StartIndex, w.EndIndex, w.AnalysisFps, w.RespirationRate, w.RangeBin,
               w.{REVIEWER_FLIPPED}, w.SystemVersion,
               COUNT(r.ID)                 AS Spans,
               COUNT(DISTINCT r.LabelerID) AS Labelers
        FROM {t['window']} w
          LEFT JOIN {t['phase_record']} r ON r.RespirationWindowID = w.ID
        GROUP BY w.ID
        ORDER BY w.RadarSignalID, w.WindowIndex
    """)


def signals_frame(env_key: str, signal_ids: list[int]) -> pd.DataFrame:
    """Patient, session and start time for the given signals. Device side only.

    The patient's display name is per environment: the algo instance carries a study name on a
    `Patient` table in its *labels* database, while production has no name column at all -
    `Session.PatientID` is the identity there.
    """
    columns = ["RadarSignalID", "SessionID", "StartTime", "Fps", "PatientKey", "Patient"]
    if not signal_ids:
        return pd.DataFrame(columns=columns)

    cfg, t = env_config(env_key), config()["tables"]
    device = cfg[DEVICE]
    ids = ", ".join(str(int(value)) for value in signal_ids)
    on_device = (device.get("patient_name_column")
                 and device.get("patient_table_in") == DEVICE)
    found = frame(env_key, DEVICE, f"""
        SELECT rs.ID AS RadarSignalID, rs.SessionID, rs.StartTime, rs.Fps,
               s.PatientID AS PatientKey,
               {f"p.{device['patient_name_column']}" if on_device else "NULL"} AS PatientName
        FROM {t['radar_signal']} rs
          JOIN {t['session']} s ON s.ID = rs.SessionID
          {f"LEFT JOIN {t['patient']} p ON p.ID = s.PatientID" if on_device else ""}
        WHERE rs.ID IN ({ids})
    """)
    if found.empty:
        return pd.DataFrame(columns=columns)

    if (device.get("patient_name_column")
            and device.get("patient_table_in") == LABELS):
        keys = sorted({str(key) for key in found["PatientKey"].dropna()})
        if keys:
            listed = ", ".join(f"'{key}'" for key in keys)
            names = frame(env_key, LABELS,
                          f"SELECT ID AS PatientKey, "
                          f"{device['patient_name_column']} AS PatientName "
                          f"FROM {t['patient']} WHERE ID IN ({listed})")
            found = found.drop(columns=["PatientName"]).merge(names, on="PatientKey", how="left")

    found["Patient"] = found["PatientName"].where(found["PatientName"].notna(),
                                                  found["PatientKey"])
    return found.drop(columns=["PatientName"])


def catalogue(env_key: str) -> pd.DataFrame:
    """Windows merged with their signal and patient. One row per window.

    This merge **is** the association between a stored window and where it came from, and it
    happens in pandas rather than SQL because on prod the two sides are different servers.
    """
    windows = windows_frame(env_key)
    if windows.empty:
        for column in ("SessionID", "StartTime", "Fps", "PatientKey", "Patient"):
            windows[column] = None
        return windows
    signals = signals_frame(env_key, sorted({int(i) for i in windows["RadarSignalID"]}))
    merged = windows.merge(signals, on="RadarSignalID", how="left")
    merged["Spans"] = merged["Spans"].fillna(0).astype(int)
    return merged


def phase_vocabulary(env_key: str) -> dict[int, str]:
    """`BreathPhaseTypeID` -> the app's word for it, lowercased to match `phase.labels`."""
    rows = frame(env_key, LABELS, f"SELECT ID, Name FROM {config()['tables']['phase_type']}")
    return {int(row.ID): str(row.Name).lower() for row in rows.itertuples()}


def human_spans(env_key: str, window_id: int) -> pd.DataFrame:
    """Every labelled span on one window, whoever made it.

    Note `EndIndex` here is **inclusive** - adjacent spans share a boundary sample - while a
    window's `EndIndex` is exclusive. Off by one shifts every human boundary.
    """
    t = config()["tables"]
    return frame(env_key, LABELS, f"""
        SELECT r.ID, r.RespirationWindowID, r.BreathPhaseTypeID, r.StartIndex, r.EndIndex,
               r.LabelerID, r.RecordCreationTime
        FROM {t['phase_record']} r
        WHERE r.RespirationWindowID = %(window)s
        ORDER BY r.LabelerID, r.StartIndex
    """, window=int(window_id))


# ------------------------------------------------------------------------------------ samples

@lru_cache(maxsize=1)
def _s3():
    from holosis_aws_manager import S3Manager

    use_profile()
    cfg = config()["s3"]
    return S3Manager(region=config()["aws"]["region"], bucket_name=cfg["bucket"])


def window_samples(s3_path: str) -> tuple[np.ndarray, np.ndarray]:
    """`(t_sec, values)` for one window, exactly as stored.

    Never re-oriented here - orientation is a build decision, see `LabelSource.orient`. The blob
    is already the trace the labelling app shows and `ReviewerFlipped` describes *that object*.
    """
    import io

    cfg = config()["s3"]
    body = _s3().s3_client.get_object(Bucket=cfg["bucket"], Key=str(s3_path))["Body"].read()
    blob = np.load(io.BytesIO(body))
    return (np.asarray(blob[cfg["blob_time_key"]], dtype=np.float32),
            np.asarray(blob[cfg["blob_values_key"]], dtype=np.float32))
