# Assets

This directory is reserved for demo assets. The CLI writes its generated
outputs (JSONL and self-contained HTML) to the directory passed via
`--out-dir`, which defaults to `demo-out/` and is git-ignored.

No pre-rendered or real-sensor assets are shipped: every dataset produced by the
`perception-cli` is synthetic and generated deterministically from a seed.
