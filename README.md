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
| [`simplejev-mars`](../../tree/simplejev-mars) | MARS RAID — same demo via [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) (Qwen 27B on llama.cpp) |
| [`gliner-mars`](../../tree/gliner-mars) | MARS RAID — [GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) (194M) on the CPU |
| [`gliner-email`](../../tree/gliner-email) | Inbox triage — GLiNER 2.5 (194M) on the CPU, 18.6 emails/s |
| [`laya-mars`](../../tree/laya-mars) | MARS RAID — [Laya](https://huggingface.co/convaiinnovations/laya) (421M) on the GPU, 4.7 ms/question |
| [`laya-email`](../../tree/laya-email) | Inbox triage — Laya (421M) on the GPU, fastest of three and worst at the job |
| [`nimble-mars`](../../tree/nimble-mars) | MARS RAID — [Bespoke Nimble 9B](https://github.com/bespokelabsai/nimble) on the GPU, one prefill per decision, 132 ms |
| [`verdict-mars`](../../tree/verdict-mars) | MARS RAID — [openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) (151M ModernBERT), non-autoregressive, 18 ms |
| [`laya-snake`](../../tree/laya-snake) | Snake — [laya-mlx](https://github.com/mizorewww/laya-mlx) (322M multilingual, MLX) playing Snake locally on Apple Silicon, with browser view |
| [`laya-coreml`](../../tree/laya-coreml) | Tetris — [Laya](https://huggingface.co/convaiinnovations/laya) (322M multilingual, Core ML) playing Tetris in the browser on Apple Silicon, 9.4 ms/decision |

**Not a GitHub fork.** The original lives only on Hugging Face. Root commit `63ab36c` vendors his `code/`
verbatim at `8c9db06`, so `git show 63ab36c` is the pristine upstream and everything after it on each branch
is ours. MIT, like his.
