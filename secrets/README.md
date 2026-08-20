# Local credentials

Git-ignored, except this file. **Never commit a password.**

Only one file is ever read from here, and only as a fallback:

| File | What it is |
| --- | --- |
| `prod_edge_ro_password.txt` | One line: the password for `edge_data_user_ro`, production's read-only database account. |

It is a fallback because the password's home is Secrets Manager
(`sm-prod-01-clinical-dashboard-edge-ro`, read under the `holosis-prod-admin` profile), and that
is what `phase/sources.py` tries first. Put it here only if you have no Secrets Manager access in
the production account. `INHALE_EXHALE_TRAINING_PROD_RO_PASSWORD` does the same job without a
file, and takes priority over both.

The account is granted `SELECT` and nothing else, so production cannot be written through it
whoever holds the credential - and nothing in this repo writes anywhere at all.
