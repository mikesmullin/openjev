# gliner-mars

[GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) playing **MARS RAID**, on the CPU, at 23 ms per
question.

Fourth model through this harness and the first one where inference is not the bottleneck. GLiNER2.5 is
a 194M DeBERTa-v3 information-extraction encoder: it scores supplied labels against text rather than
generating anything, and `classify_text` answers a dict of independent tasks in one forward pass —
which is the shape this harness has wanted since the first branch.

| branch | model | how it decides | per decision | device |
|---|---|---|---|---|
| [`openjev-mars`](../../tree/openjev-mars) | Qwen3.5-4B NLI cross-encoder | scores hypotheses | ~35 ms | GPU |
| [`simplejev-mars`](../../tree/simplejev-mars) | Qwen3.8-27B via llama.cpp | reads label logits | ~430 ms | GPU |
| *(deleted)* DiffusionGemma | 26B-A4B text diffusion | generates JSON | ~3700 ms | GPU + offload |
| **`gliner-mars`** *(here)* | GLiNER 2.5, 194M | classifies labels | **~71 ms** | **CPU** |

This branch is an orphan — it shares no history with the others.

## Results

Measured in the browser, one RTT per decision, three questions answered per inference:

| | |
|---|---|
| RTT | **p50 71 ms**, mean 76 ms, p95 88 ms |
| per question | **23 ms** (3 questions in one forward pass) |
| proxy overhead | 4 ms |
| model load | 6.3 s |
| device | CPU, 12 threads — the GPU is never touched |

Model comparison on the same four-question payload, 12 CPU threads:

| model | params | 4 questions | per question |
|---|---|---|---|
| `fastino/gliner2.5-base-v1` | 194M | 68.1 ms | 17.0 ms |
| `fastino/gliner2.5-small-v1` | 74M | 33.1 ms | 8.3 ms |

Decision quality after tuning, verified across the states that matter:

```
hull=100  saucers=0   bldgs=11  ->  threat 1  target building  press      wake no
hull=50   saucers=6   bldgs=5   ->  threat 2  target saucer    press      wake no
hull=15   saucers=12  bldgs=2   ->  threat 3  target saucer    break_off  wake no
hull=90   saucers=0   bldgs=0   ->  threat 1  target building  press      wake yes
hull=70   colony flat, boss up  ->  target the scorpion's claw, over the saucers
```

The **survival** card on the page is the score that compares across branches: every other number
measures the model, that one measures whether the decisions were any good.

## What GLiNER can and cannot be asked

### It cannot read numbers

This is the single most important thing to know before wiring it to anything numeric. Same situation,
two framings, `posture` over `["attack", "retreat"]`:

```
"Hull 15 percent. lost 25 hull in the last ten seconds."      -> attack  (0.99)   WRONG
"critically damaged and almost destroyed, must escape now"    -> retreat (1.00)   right
```

A DeBERTa encoder scoring labels has no arithmetic: `15 percent` is a token, not a quantity, and it
cannot be told apart from `95 percent`. Given the same situation in words it is decisive and correct.
So `describe()` crosses the numeric bands in code and hands the model prose. **The thresholds are ours
and the judgement is the model's** — worth stating plainly rather than dressing up.

### Posture is systematically inverted

Asked to choose between attacking and retreating, it answers with total confidence and exactly
backwards, across three separate wordings on the same states:

```
["attack", "retreat"]                                  safe -> retreat    (1.00)   dying -> attack (1.00)
["keep attacking", "break off and escape"]             safe -> break off  (0.91)   dying -> attack (1.00)
["press the attack on the colony", "flee to survive"]  safe -> flee       (1.00)   dying -> press  (0.99)
```

Consistent inversion at high confidence is not noise: "attack" matches a text full of saucers and
damage, which is precisely when the ship should be leaving. Inverting the result would be cargo-culting
something we do not understand, so **posture is derived from the model's own `threat` score**, which is
monotonic and correct. The page labels it *derived from threat · not asked of the model*.

### Labels want to be class names, not sentences

The first version used a full sentence per label ("the ship is about to be destroyed"). `threat` read
maximal on every tick, including at 100 percent hull with an empty sky. Short, distinct class names
(`safe` / `minor damage` / `heavy damage` / `critical`) fixed it. This is the previous branches' lesson
pointing the other way: a generative model wants an option that argues for itself, a classification
encoder wants a label that names a class.

`wake` is the exception that proves it — there the two options had to state the *reason*, because the
bare actions are not distinguishable from the text:

```
["wake the scorpion", "leave it buried"]                     always "leave it buried"
["wake the sleeping scorpion", "keep destroying the colony"] always "wake"
["the colony is finished so wake the scorpion",
 "there are still buildings to destroy"]                     right, and 0.99 on the case that matters
```

### Targets are classified by kind, not by instance

Separating "the building 137 metres away" from "the building 174 metres away" is asking a text
classifier to compare two digits. The model picks *what kind of thing* to attack; the code picks the
nearest one of that kind from a list it has already sorted. Geometry stays in the code, judgement stays
in the model — the same split the whole harness is built on, one level up.

## Priority by structure, not by wording

Saucers respawn forever, so clearing them is not a win condition — the mission is the colony and then
the scorpion. Offered unconditionally, GLiNER picked "alien saucer" on essentially every tick that one
existed, because the state text called them *in the air shooting at it*, the most urgent-sounding
phrase in the paragraph.

This is the openjev lesson arriving from the opposite direction. There the saucer option argued against
itself and scored ~0.01 while the ship was shot down; the fix was to make it compelling when the danger
was real. Here it was compelling always. The principle is the same either way: **an option belongs on
the ballot only when choosing it would be right.** Saucers are now offered only once they are actually
doing damage, so below that threshold the model cannot pick them because it is not asked to.

Two wording fixes fell out of it:

- The saucer phrase tracks what they are doing — *"buzzing around as a harmless distraction"* when they
  are missing, *"shooting the ship apart and have to be cleared first"* when they are not. One fixed
  phrase either won every tick or lost every tick.
- A mission sentence was tried and **removed**. *"The mission is to flatten the colony and then kill the
  giant scorpion"* is true on every tick, and it made `target` pick the colony even while the ship was
  being destroyed. That is the third time in this repo an always-true sentence has quietly decided an
  answer, after openjev's *"clearing them is endless"* and this branch's *"The ship dies at zero hull."*

## A 70 ms model does not mean a 70 ms game

Worth recording because it cost several wrong turns. With decisions this cheap the obvious move is to
run the loop as fast as the model allows. At ~4 decisions/second the ship fired **2 shots in 28 seconds**
with eleven buildings standing: the aim servo needs a few hundred milliseconds to bring the nose inside
its 4.5 degree firing threshold, and it never got them.

Two "fixes" were tried and both made it worse — committing to a target until its kind changes, and
suppressing the `aim()` call unless the chosen uid changed. The autopilot is written to be told every
tick and reads `target.live()` each frame, so re-seating it is the normal case, not a disturbance. What
actually helped was slowing the loop to 900 ms, which is roughly what the earlier branches used.

**The model finishing in 70 ms of a 900 ms budget is the result.** Spending the other 830 ms letting the
controller execute is not a compromise.

### The frame-rate trap

One debugging note that wasted more time than anything else. Driving the page over CDP, the game ran at
**1.7 FPS** while `document.hidden` was `false`. The agent loop is `setTimeout`-driven so it kept perfect
time and the HUD looked entirely normal — 34 decisions, 71 ms p50 — while the sim underneath rendered
about one frame per second. Every symptom pointed at targeting: aim oscillating between 0.6° and 11.4°,
`firing: true` with three shots landed, no saucers ever spawning.

A controller that is aimed and firing but not hitting is a *time* problem, not a targeting problem. The
survival clock on the page now runs off wall time on its own interval for exactly this reason — a timer
derived from decision ticks would have read normally throughout.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "gliner2[local]" protobuf sentencepiece
bun run game                     # fetch + patch vibe-arcade's mars.html

bun run model                    # terminal 1: the classifier, CPU, port 8780
bun run dev                      # terminal 2: http://127.0.0.1:8734/
```

`protobuf` and `sentencepiece` are not optional — DeBERTa-v3's tokenizer needs both, and
`from_pretrained` fails with an `ImportError` without them.

`bun run model` starts under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`. The model
is 194M and comes nowhere near that, but a kernel-enforced ceiling means a runaway load kills its own
process rather than the desktop.

## Architecture

```
browser  web/index.html + web/app.js      m.js page; the agent loop lives here, next to the game
         web/game/mars.html  (iframe)     vibe-arcade's game + one injected bridge line
         web/game/mars-hook.js            state -> candidates, autopilot, aiming
   |
   v  POST /api/decide
bun      server/static.js                 static files + proxy. No game logic.
   |
   v  POST /v1/decide
python   server/gliner_server.py          ~200 lines. fastino/gliner2.5-base-v1 on the CPU.
```

## Honest placement

Only `target` and `threat` are really the model's: `posture` is a threshold on `threat`, the numeric
bands are code, and the choice between instances of a kind is distance. What GLiNER contributes is the
part it is built for — reading a described situation and naming which category of thing matters right
now — and it does that in 23 ms per question on a CPU, which is the whole point.

`classify_text` returns the winning label and its confidence, not a distribution: `format_results=False`,
`threshold=0.0` and multi-label mode all still return just the argmax. So the page's per-candidate bars
show one full bar rather than a spread. openjev and simple-jev remain the only branches that could rank
the whole ballot.
