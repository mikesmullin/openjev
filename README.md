# openjev

[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) — Qwen3.5-4B fine-tuned as a 3-class NLI
cross-encoder — reproduced locally, then pointed at games and real work. The model never generates text. It
scores statements and the argmax entailment is the answer.

This branch is an index only. The work lives on the topic branches:

| branch | task |
|---|---|
| [`openjev-doom`](../../tree/openjev-doom) | ViZDoom — the original reproduction |
| [`openjev-cook`](../../tree/openjev-cook) | Cook Fever (vibe-arcade `cook2.html`) in the browser |
| [`openjev-tetris`](../../tree/openjev-tetris) | Tetris ([mikesmullin/tetris](https://github.com/mikesmullin/tetris)) in the terminal |
| [`openjev-mars`](../../tree/openjev-mars) | MARS RAID (vibe-arcade `mars.html`) — target selection in a 3D shooter |
| [`openjev-email`](../../tree/openjev-email) | Inbox triage — action recommendation plus a spam score per email |

**Not a GitHub fork.** The original lives only on Hugging Face. Root commit `63ab36c` vendors his `code/`
verbatim at `8c9db06`, so `git show 63ab36c` is the pristine upstream and everything after it on each branch
is ours. MIT, like his.
