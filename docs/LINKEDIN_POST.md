# LinkedIn — two posts

Post 1 (video) goes first. Post 2 (graphs) follows 2-3 days later.

**Paste-ready below.** Each paragraph is a single unbroken line — LinkedIn does
its own wrapping, so do not reflow these or the line breaks will show up in the
post. No markdown: LinkedIn renders `**bold**` literally, so emphasis is done
with sentence structure instead.

---

## POST 1 — upload as a VIDEO post

Media: `out/2dvideogen_reel.mp4` (44 s, 1.1 MB) — the three clips with title cards.

```text
Can you fine-tune a small model to generate 2D video? Not a big one. A small one, on a laptop.

That was the whole experiment. Here's how far it got.

No custom architecture, nothing trained from scratch. Just t5-small — 60M parameters, an ordinary off-the-shelf text model — fine-tuned to write animation scripts instead of sentences. Who's on stage, where they stand, what they do, when. A renderer turns that script into the mp4. The model never touches a pixel.

I should say this up front: most of the code here was written by Claude Code, not by me. The training and eval harnesses, the renderer, the data generator. My part was deciding what to try, designing the experiments, and checking every number that came back — which, as it turned out, was the part that actually mattered. It also caught a silent GPU grab early on that would have cooked my laptop's already-broken cooling fan.

On data: nothing scraped. The model trained on 6,000 synthetic prompt/script pairs generated on my own machine. The diffusion route I tried first used public-domain footage only — COCO photos for pose extraction, Muybridge's 1887 motion studies for how people actually walk, and lapsed-copyright Fleischer cartoons (Superman, Betty Boop) for style. Licensing was a filter from the start, not something sorted out afterwards.

Why bother, when Sora and Veo do this far better? They need a datacenter; this trains in nine minutes on a CPU. And their output is final, where mine is a text file you can edit — a character in the wrong place is a number on line 9, not a reroll.

Then I tried the obvious upgrade: same data, same settings, a model 3.7× bigger. 220M instead of 60M.

Every benchmark went up. Prompt accuracy 46% to 57%. Valid output 94% to 99%.

The video looked exactly the same. The last two clips in the reel are the two model sizes on the same prompt, and I genuinely cannot tell them apart.

So — a fine-tuned 60M model can generate coherent 2D animation from a sentence. Rough, stick figures, nowhere near the big systems. But it runs on hardware you already own, which was the whole question.

What's next: training something from scratch for actual anime output, sized deliberately to fit in small VRAM. Fine-tuning a text model got me further than I expected it would. I think the ceiling here is the architecture, not the effort.

All of it is public — 24 attempts, roughly half of them failures, each one with the real reason it died.

github.com/SahilSidhu7/2DVideoGen
```

---

## POST 2 — upload as an IMAGE post, 2-3 days later

Media, in order: `docs/media/bench_loss_vs_quality.png`,
`docs/media/bench_staging_decode.png`, `docs/media/bench_60m_vs_220m.png`

```text
Last week I posted about fine-tuning a 60M model to generate 2D animation on a laptop, and said plainly that Claude Code wrote most of the code.

So here's what was actually left for me to do: the measuring. And my measurements lied to me three separate times.

1. My loss function was wrong for three checkpoints in a row.

First graph. Loss went up — 0.924, then 0.947, then 1.004 — while real accuracy on unseen prompts went 25%, 62%, 71%. Each dataset I generated was harder than the last, so a worse loss meant a better model every single time. If I'd trusted the number on the screen I'd have shipped the worst of the four and felt good about it.

2. The model "couldn't count."

Ask it for six characters, get three. I assumed it was too small and started planning a bigger one. Then I actually looked at the training data: in 5,681 of 5,688 examples, the number of phrases in the sentence happened to equal the number of characters. It had learned to count phrases. It had never once been asked to read the number. The fix was in the generator, not the model, and the gap fell from 66 points to 24. Synthetic data doesn't mean clean data.

3. The worst bug wasn't a bug.

Characters kept piling onto the same spot. I logged it as the model failing to learn staging. It was beam search doing exactly its job — reporting the single most likely answer from a distribution that was already fine. Same weights, switched to sampling: 0.19 to 0.86. Second graph.

Three times I was one step away from fixing the wrong thing. Each one got caught the same way — a number looked too clean, and I went back to check it.

That's the part the AI assistance didn't cover. Claude wrote most of the code and found real bugs in it. It could not tell me that my metric was measuring the wrong thing.

Which is roughly where I've landed: writing the code stopped being the bottleneck a while ago. Knowing what to measure, and refusing to believe a clean result, hasn't.

Has a metric ever confidently lied to you?

github.com/SahilSidhu7/2DVideoGen
```
