# No local credentials

This directory is git-ignored and **nothing here is read any more**. Every credential - host,
user and password - comes out of Secrets Manager, under the secret named for that connection in
`parameter/sources/default.yaml`. There is no password file and no environment variable, and a
test fails the build if a `password:` key ever appears in a tracked file.
