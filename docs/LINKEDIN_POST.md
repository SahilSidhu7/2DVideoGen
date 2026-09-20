# LinkedIn — two posts

Post 1 goes up first (video). Post 2 follows 2–3 days later (graphs), and
links back to post 1.

---

## POST 1 — the story (upload as a VIDEO post)

**Media:** one video — `out/2dvideogen_reel.mp4` (44 s, 1.1 MB).
All three clips stitched with title cards: Attempt 22 → Attempt 23 →
Attempt 24 → closing card. Built by `out/stitch/`; re-runnable.

---

I tripled the size of my model to make this animation better.

It came out exactly the same.

---

For a few months I've been trying to get a laptop to turn a sentence into 2D
animation. Type "two friends meet in the park, one waves, then they kick a ball
around" — get an actual mp4.

The constraint that shaped everything: one laptop, an 8GB GPU, a broken cooling
fan. Every GPU run throttled to 50%, stopping if the chip hits 70°C.

That killed my first plan — Stable Diffusion frames, styled anime, the obvious
route. Single frames looked gorgeous. In motion it flickered 25× worse than my
pass bar, and the module meant to smooth it made the characters stop following
their pose guides entirely. Two fixes that should have stacked cancelled out.

So I gave up on generating pixels and generated *instructions* instead. A small
model writes a scene script — who's on stage, where they stand, what they do and
when — and a renderer I wrote draws it. 60M parameters. Trains in nine minutes
on a CPU.

Worth saying plainly: **this problem is already solved commercially.** Sora,
Veo, Runway will hand you a better-looking clip from the same sentence, today.
What they won't hand you is something you can edit. When their model puts a
character in the wrong spot, there's nothing to fix. Mine writes a text file
where the wrong spot is a number on line 9 — and re-renders identically every
time. That was the bet: give up quality, buy control.

That worked. So I did what everyone does when something works but feels small:
made the model 3.7× bigger. Same data, same settings.

Every benchmark went up. Prompt accuracy 46% → 57%. Valid output 94% → 99%.

Then I put the frames side by side and couldn't tell them apart. The bigger one
was smoother, the smaller one moved more, both stacked two characters on top of
each other, and the new one had a character kick a ball at herself.

Months of work, one honest summary: **the bottleneck was never model size.**
Cleaning up my training data helped more. Changing *one decoder setting* helped
more than that.

Public, failures and all — 24 attempts, half of them dead ends, each with the
real reason it died.

🔗 github.com/SahilSidhu7/2DVideoGen

---

## POST 2 — the graphs (upload as an IMAGE post, 2–3 days later)

**Media, in order:**
1. `docs/media/bench_loss_vs_quality.png`
2. `docs/media/bench_staging_decode.png`
3. `docs/media/bench_60m_vs_220m.png`

---

Last week I posted about an animation model that didn't get better when I made
it bigger. Here's the part that actually cost me time: three separate occasions
where my own measurements confidently lied to me.

**1. My loss function was wrong for three checkpoints straight.**
Graph 1. Loss went *up* — 0.924 → 0.947 → 1.004 — while real accuracy went
25% → 62% → 71%. Each dataset was harder than the last, so worse loss meant a
better model every single time. If I'd trusted the number on the screen, I'd
have shipped the worst of the four.

**2. The model "couldn't count."**
Ask for six characters, get three. I assumed a capacity limit and started
planning a bigger model. Then I checked the data: in 5,681 of 5,688 training
examples, the number of phrases in the sentence happened to equal the number of
characters. It had learned to count phrases. It had never once been asked to
read the number. I fixed the data — not the model — and the gap dropped from 66
points to 24.

**3. The worst bug wasn't a bug.**
Characters kept piling up on the same spot. I wrote it up as a failure to learn
staging. It was beam search doing its job: reporting the single most likely
answer from a distribution that was already fine. Same weights, switched to
sampling — 0.19 → 0.86. Graph 2.

Three times I was about to fix the wrong thing. Each one was caught by
distrusting a number that looked too clean.

I used Claude Code throughout — harnesses, debugging, write-ups. It caught a
silent GPU grab that would have cooked my broken fan. It also couldn't have told
me any of the three things above; those came from re-running the measurement
while suspicious.

Has a metric ever confidently lied to you? I'd like to hear it.

🔗 github.com/SahilSidhu7/2DVideoGen
