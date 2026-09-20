# LinkedIn — two posts

Post 1 (video) goes first. Post 2 (graphs) follows 2–3 days later and links back.

---

## POST 1 — the experiment (upload as a VIDEO post)

**Media:** `out/2dvideogen_reel.mp4` — 44 s, 1.1 MB. Three clips with title
cards: Attempt 22 → Attempt 23 → Attempt 24 → closing card.

---

Can you fine-tune a *small* model to generate 2D video? Not a big one. A small
one, on a laptop.

That was the whole experiment. Here's how far it got.

---

No custom architecture, no training from scratch. I took t5-small — 60M
parameters, an ordinary off-the-shelf model — and fine-tuned it to write
**animation scripts** instead of text. Who's on stage, where they stand, what
they do, when. A renderer turns that script into an mp4.

That's it. That's the trick. The model never touches a pixel.

**The training data is the part people ask about.** There isn't a scraped
dataset behind this. I wrote a generator that produces prompt/script pairs and
trained on **6,000 of them** — synthetic, made on my own machine. For the
diffusion route I tried first, I used only public-domain footage: COCO photos
for pose extraction, Eadweard Muybridge's 1887 motion studies for gait, and
lapsed-copyright Fleischer cartoons (Superman, Betty Boop) for animation style.
Licensing was a selection rule, not an afterthought. Full provenance is in the
repo.

Why bother, when Sora and Veo do this far better? They need a datacenter; this
trains in **nine minutes on a CPU**. And their output is final, while mine is a
text file you can edit — a character in the wrong spot is a number on line 9,
not a reroll.

**I used Claude Code for most of the build** — harnesses, renderer, debugging.
It caught a silent GPU grab that would have cooked my laptop's already-broken
cooling fan. Pair programming, where I set direction and checked measurements.

Then I tried the obvious upgrade: same data, same settings, a 3.7× bigger model.
220M instead of 60M.

Every benchmark went up. Prompt accuracy 46% → 57%. Valid output 94% → 99%.

The video looked exactly the same. Watch the reel — the last two clips are the
two model sizes, same prompt. I can't tell them apart either.

So: a fine-tuned 60M model *can* generate coherent 2D animation from a sentence.
Rough, stick figures, nowhere near the big systems — but it runs on hardware you
already own.

**Next:** training something from scratch for actual anime output, sized to fit
small VRAM. Fine-tuning a text model got me further than expected. I think the
ceiling here is architecture, not effort.

All public — 24 attempts, half of them failures, each with the real reason.

🔗 github.com/SahilSidhu7/2DVideoGen

---

## POST 2 — what went wrong (upload as an IMAGE post, 2–3 days later)

**Media, in order:**
1. `docs/media/bench_loss_vs_quality.png`
2. `docs/media/bench_staging_decode.png`
3. `docs/media/bench_60m_vs_220m.png`

---

Last week I posted about fine-tuning a 60M model to generate 2D animation on a
laptop. The build was the easy half — I had Claude Code for that.

The hard half was that my own measurements lied to me three times.

**1. My loss function was wrong for three checkpoints straight.**
Graph 1. Loss went *up* — 0.924 → 0.947 → 1.004 — while real accuracy went
25% → 62% → 71%. Each dataset was harder than the last, so worse loss meant a
better model every time. Trusting the number on screen would have shipped the
worst of the four.

**2. The model "couldn't count."**
Ask for six characters, get three. I assumed it was too small and started
planning a bigger one. Then I checked the data I'd generated myself: in 5,681 of
5,688 examples, the number of phrases in the sentence happened to equal the
number of characters. It had learned to count phrases. It was never once asked
to read the number. I fixed my generator, not the model — gap dropped from 66
points to 24. Synthetic data doesn't mean clean data.

**3. The worst bug wasn't a bug.**
Characters kept piling onto the same spot. I wrote it up as a failure to learn
staging. It was beam search doing its job — reporting the single most likely
answer from a distribution that was already fine. Same weights, switched to
sampling: 0.19 → 0.86. Graph 2.

Three times I was one step from fixing the wrong thing. Each was caught by
re-running a measurement I didn't believe.

That's the part AI assistance doesn't cover. Claude wrote most of my code and
caught real bugs in it. It could not have told me my metric was measuring the
wrong thing — that only showed up because a number looked too clean and I went
back to check.

Has a metric ever confidently lied to you?

🔗 github.com/SahilSidhu7/2DVideoGen
