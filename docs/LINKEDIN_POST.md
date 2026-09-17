# LinkedIn post — 2DVideoGen

**Media order (LinkedIn carousel / multi-image post):**

1. `docs/media/bench_loss_vs_quality.png` — the hook graph
2. `docs/media/a20_anime_diffusion.gif` — the pretty failure (Attempt 20)
3. `docs/media/a22_generated.gif` — first end-to-end generated clip (Attempt 22)
4. `docs/media/a24_six.gif` — the 220M model, six characters (Attempt 24)
5. `docs/media/bench_60m_vs_220m.png` — 60M vs 220M
6. `docs/media/bench_staging_decode.png` — the decoder finding

---

## Post

I spent 24 attempts trying to make a laptop generate 2D animation from a
sentence. Then I tripled the model size to fix it.

Every benchmark improved. The video looked exactly the same.

That's the part I want to talk about.

---

**The setup:** one laptop. 8GB GPU with a broken cooling fan, so every GPU run
is throttled to 50% duty by a script that watches the die temperature. Half of
the 24 attempts failed. I logged all of them.

**What I built:** a model that reads "two friends meet in the park, one waves,
then they kick a ball around" and writes a scene script — cast, positions,
colours, a timeline of events — which a renderer turns into an actual mp4.
60M parameters. Trained in 548 seconds. On CPU.

**What I tried first, and killed:** the obvious route — Stable Diffusion +
ControlNet + AnimateDiff, styled anime frames. Slide 2 is one frame of it. It
looks good. It's also a still, because in motion it flickered 25× over the pass
bar, and adding the motion module made the character *stop following the pose
guide entirely*. Two objectives that were supposed to compound interfered
instead. I closed that route by measurement and wrote down why.

**Then the scaling test.** The pipeline worked but felt small, so I did the
thing everyone does: same data, same hyperparameters, 3.7× the parameters.
60M → 220M.

The results (slide 5):
→ prompt accuracy 46% → 57%
→ actions per character 73% → 89%
→ structurally valid without repair 94% → 99%
→ staging quality 0.879 → 0.911

Real gains, all of them. Then I rendered the same scene with both models and
put the frames side by side.

A tie. The bigger model was smoother; the smaller one had more movement. Both
made the same mistake — two characters standing on top of each other — just in
a different place on screen. The new one also had a character kick a ball at
herself.

**Four things this taught me, and the first three cost me the most time:**

**1. My loss function lied to me for three checkpoints in a row.** Slide 1.
Loss went UP — 0.924 → 0.947 → 1.004 — while real-world accuracy went
25% → 62% → 71%. Each new dataset was harder than the last, so "worse" loss
meant a better model every time. If I'd optimised the number on the screen I'd
have shipped the worst model of the four.

**2. A model can look incapable when the data never asked.** Mine seemed unable
to count a cast of six. It wasn't. In 5,681 of 5,688 training examples, the
number of clauses in the sentence happened to equal the number of characters.
It had learned to count clauses. It had never once been asked to read the
numeral. I fixed the data, not the model, and the generalisation gap fell from
66 points to 24.

**3. The bug wasn't in the model. It was in one decoder setting.** Characters
kept stacking on top of each other. I wrote it up as a model failure. It was
beam search reporting the most likely answer from a distribution that was
already fine. Same weights, switched to sampling: staging 0.19 → 0.86.
(Slide 6 — and notice the 220M model doesn't need the trick at all.)

**4. Capacity was never the bottleneck.** Deconfounding a dataset and changing
a decoder flag moved this project further than tripling the model did. The
scale-up bought robustness — near-perfect structural validity, and it removed a
tradeoff I'd recorded as permanent. It did not buy a better-looking video.

**What it still can't do:** it does not understand "stand on the left." I
supervised it, measured it, and got 2.7 points over a control — noise. And the
best-looking clip in the whole project is still the one I wrote by hand.

So: a failure, by the goal I set. But four findings I'd never have gotten from
a project that worked on the first try — and all of them are about *the
measurement*, not the model.

I used Claude Code throughout — building the training and eval harnesses,
catching a CUDA OOM and a silent GPU grab that would have cooked my broken fan,
and drafting the write-ups. It wrote code and prose. Every number here came out
of a script that actually ran.

Everything is public: 24 attempts, the failures with their real causes, the
retractions, and the four times my own evaluator gave me a confident wrong
answer.

🔗 github.com/SahilSidhu7/2DVideoGen

---

If you've had a metric confidently lie to you, I'd like to hear it. I'm fairly
sure mine isn't done.

#MachineLearning #AI #ComputerVision #DeepLearning #BuildInPublic
