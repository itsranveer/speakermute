# Security policy

speakermute is a local command-line tool. It runs ffmpeg and WhisperX on files you point it at and
talks to the network only to download model weights from Hugging Face. There is no server component
and no account.

## Reporting a vulnerability

If you find a security problem (for example a crafted file name or transcript that lets a
subprocess run something it should not), please do **not** open a public issue. Email
<ranveersinghbais@gmail.com> with the details and, if you can, a reproducible example. You will get a
reply within a week. Once a fix is released the issue will be described in the changelog, with credit
if you want it.

## Supported versions

Only the latest release on the `main` branch receives fixes. There is no bug bounty.
