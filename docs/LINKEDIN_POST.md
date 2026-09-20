# LinkedIn — two posts

Post 1 goes up first (video). Post 2 follows 2–3 days later (graphs), and
links back to post 1.

---

## POST 1 — the story (upload as a VIDEO post)

**Media:** upload the three mp4s so they autoplay.
`out/a22_generated.mp4` → `out/a23_six.mp4` → `out/a24_six.mp4`
(LinkedIn allows one video per post — if you want all three, stitch them
with a 1-second title card each, or lead with `out/a24_six.mp4` alone.)

---

I tripled the size of my model to make this animation better.

It came out exactly the same.

---

For a few months I've been trying to get a laptop to turn a sentence into 2D
animation. Type "two friends meet in the park, one waves, then they kick a ball
around" — get an actual mp4.

The constraint that shaped everything: one laptop, an 8GB GPU, and a broken
cooling fan. Every GPU run is throttled to 50% and stops if the chip hits 70°C.
That single fact killed my first plan — Stable Diffusion frames, styled anime,
the obvious route. Individual frames looked gorgeous. In motion it flickered 25×
worse than the bar, and when I added the module that was supposed to smooth it,
the characters stopped following their pose guides entirely. Two fixes that were
meant to stack cancelled each other out instead.

So I gave up on generating pixels and generated *instructions* instead. A small
model writes a scene script — who's on stage, where they stand, what they do and
when — and a renderer I wrote draws it. 60M parameters. Trains in nine minutes
on a CPU.

That worked. Then I did what everyone does when something works but feels small:
made the model 3.7× bigger. Same data, same settings, more parameters.

Every benchmark went up. Prompt accuracy 46% → 57%. Valid output 94% → 99%.

Then I rendered the same scene with both models, put the frames side by side,
and couldn't tell them apart. The bigger one was smoother. The smaller one moved
more. Both stacked two characters on top of each other — just in different
places. The new one also had a character kick a ball at herself.

Months of work, and the honest summary is: **the bottleneck was never the model
size.** Cleaning up my training data helped more. Changing *one decoder setting*
helped more than that.

It's public, failures and all — 24 attempts, half of them dead ends, each with
the real reason it died.

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
