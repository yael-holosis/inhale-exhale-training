"""The blob cache. No AWS: the download is stubbed, which is the point - a hit must not reach it.

The property under test is the one that makes the cache safe: it holds the **raw blob** and
nothing derived from the database. `ReviewerFlipped` and the labels move; the blob does not.
"""

import numpy as np
import pytest

from phase import sources

TIMES = np.arange(5, dtype=np.float32) / 10
VALUES = np.array([0.0, 1.0, 2.0, 1.0, -1.0], dtype=np.float32)
KEY = "sessions/42/windows/7.npz"


@pytest.fixture
def cached(tmp_path, monkeypatch):
    """Point the cache at a temporary directory and count downloads."""
    cfg = dict(sources.config())
    cfg["s3"] = {**cfg["s3"], "cache_dir": str(tmp_path)}
    monkeypatch.setattr(sources, "config", lambda: cfg)

    calls = {"n": 0}

    class Body:
        @staticmethod
        def read():
            import io
            calls["n"] += 1
            buffer = io.BytesIO()
            np.savez(buffer, **{cfg["s3"]["blob_time_key"]: TIMES,
                                cfg["s3"]["blob_values_key"]: state["values"]})
            return buffer.getvalue()

    state = {"etag": "aaa", "values": VALUES}

    class Client:
        s3_client = type("S3", (), {
            "get_object": staticmethod(lambda **kwargs: {"Body": Body()}),
            "head_object": staticmethod(lambda **kwargs: {"ETag": f'"{state["etag"]}"'}),
        })()

    monkeypatch.setattr(sources, "_s3", lambda: Client())
    sources.CACHE_HITS = sources.CACHE_MISSES = 0
    calls["state"] = state
    return calls


def test_the_second_read_does_not_download(cached):
    first = sources.window_samples(KEY)
    second = sources.window_samples(KEY)
    assert cached["n"] == 1, "the second read must come from the cache"
    assert np.array_equal(first[1], second[1])
    assert (sources.CACHE_HITS, sources.CACHE_MISSES) == (1, 1)


def test_refresh_downloads_again(cached):
    sources.window_samples(KEY)
    sources.window_samples(KEY, refresh=True)
    assert cached["n"] == 2


def test_the_cache_holds_the_raw_blob_so_a_changed_flip_is_not_frozen_into_it(cached):
    """`ReviewerFlipped` lives in the database and can change after the blob was fetched.

    Orientation is applied on top of the cached trace at build time, so flipping the flag
    changes the next build's output without the cache having to be invalidated.
    """
    from phase.labelsources import ALGORITHM, LabelSource
    import pandas as pd

    _, values = sources.window_samples(KEY)
    row = pd.Series({"ID": 1, sources.REVIEWER_FLIPPED: True})

    off = LabelSource("ds_algo", {"source": ALGORITHM, "orient_by_reviewer_flip": False})
    on = LabelSource("ds_algo", {"source": ALGORITHM, "orient_by_reviewer_flip": True})

    # One download, both orientations available from it.
    assert cached["n"] == 1
    assert off.orient(row, values).tolist() == VALUES.tolist()
    assert on.orient(row, values).tolist() == (-VALUES).tolist()
    # And the file on disk is still the unflipped blob.
    assert np.array_equal(sources.window_samples(KEY)[1], VALUES)


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(cached):
    sources.window_samples(KEY)
    corrupt = sources.cache_path(KEY)
    corrupt.write_bytes(b"not an npz")
    _, values = sources.window_samples(KEY)
    assert np.array_equal(values, VALUES)
    assert cached["n"] == 2


def test_no_partial_files_are_left_behind(cached):
    sources.window_samples(KEY)
    root = sources.cache_root()
    assert not list(root.rglob("*.partial"))


def test_caching_can_be_turned_off(tmp_path, monkeypatch):
    cfg = dict(sources.config())
    cfg["s3"] = {**cfg["s3"], "cache_dir": None}
    monkeypatch.setattr(sources, "config", lambda: cfg)
    assert sources.cache_root() is None
    assert sources.cache_path(KEY) is None


def test_a_reflipped_blob_is_refetched_rather_than_served_from_cache(cached):
    """The labelling app turns a window over by writing the negated samples back to the SAME key
    and toggling `ReviewerFlipped`. A cache keyed on the path alone would serve the pre-flip
    trace forever, so a hit is only a hit while the ETag still matches."""
    first = sources.window_samples(KEY)[1]
    assert np.array_equal(first, VALUES)

    # The reviewer flips it: same key, negated samples, new ETag.
    cached["state"]["values"] = -VALUES
    cached["state"]["etag"] = "bbb"

    second = sources.window_samples(KEY)[1]
    assert np.array_equal(second, -VALUES), "a rewritten blob must not come from the cache"
    assert cached["n"] == 2


def test_an_unchanged_etag_still_serves_from_cache(cached):
    sources.window_samples(KEY)
    sources.window_samples(KEY)
    assert cached["n"] == 1, "an unchanged ETag must not trigger a download"


def test_every_account_a_build_touches_is_signed_in(monkeypatch):
    """`aws.profile` reaches the data-science account; an environment whose password sits in
    another account names its own `secret_profile`, and that one is not needed until the build is
    already minutes in. Signing into the first alone is what let a build start and then die."""
    monkeypatch.setattr(sources, "config", lambda: {
        "aws": {"profile": "main", "profile_env_var": "NOT_SET"},
        "environments": {
            "ds_algo": {"labels": {"secret_name": "a"}},
            "ds_prod": {"labels": {"secret_name": "b", "secret_profile": "other-account"},
                        "windows": {"secret_name": "c", "secret_profile": "other-account"}},
        }})
    # Every account, each once, the main profile first.
    assert sources.required_profiles() == ["main", "other-account"]


def test_the_real_config_needs_two_accounts():
    assert len(sources.required_profiles()) >= 2
