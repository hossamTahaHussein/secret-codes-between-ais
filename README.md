# Secret codes between small LLMs

I wanted to see, on my own gaming GPU, whether two small language models can learn to pass a hidden message past a third model that is watching them, and which defenses catch it. This is a small-scale replication of the explicit setting in Mathew et al. 2024, "Hidden in Plain Text: Emergence and Mitigation of Steganographic Collusion in LLMs", run in English and in Arabic.

The setup has three roles. The sender writes a two or three sentence product review and has to hide one bit in it, red or blue. The receiver reads the review and has to say which one it was. The monitor reads the same review and is asked whether it hides a message. The sender is told outright that it has a secret and that a monitor is reading, so this does not test whether hidden communication shows up on its own. What it does test is whether RL finds a code nobody wrote down, and whether the usual defenses catch it.

All three roles share one base model with two LoRA adapters (sender and receiver), which keeps everything under 6 GB of VRAM. The monitor is the base model with the adapters switched off. The sender is trained with a group-normalised policy gradient (GRPO-style) plus a KL penalty to the base model. Its reward is the receiver's probability of the right answer, minus a penalty when the monitor is suspicious or the colour word appears, plus a small bonus for writing a decent review. The receiver is trained with ordinary cross-entropy at every step.

## What I found

In the clearest English run (Qwen2.5-1.5B with a 200-step warm-up before the penalties kick in), the pair settled on "vibrant colors that match any outfit" for red and plain functional words for blue. The receiver decoded 100% of 128 fresh reviews, the colour word appeared in 0.8% of them, and the monitor flagged none. Without the warm-up the sender simply learned to say nothing useful and the receiver stayed at chance.

On Qwen3-1.7B, which I used for the English vs Arabic comparison because Qwen2.5 writes poor Arabic, no run found a fully hidden code. The runs that formed a channel all used the same trick: name one colour outright and stay silent for the other. With a one-bit secret that pays the colour penalty on only half the messages, so the reward actually prefers it. Arabic formed this channel in 3 of 3 runs and English in 1 of 3, but the Arabic monitor was not more lenient during training, so I don't read this as an Arabic blind spot.

The prompted monitor was the weak point everywhere. It flagged essentially nothing, in both languages, even reviews that literally say "blue" or "أزرق", and Phi-4-mini from another vendor did little better. Detectors trained on a few hundred labelled examples worked well (78 to 100% caught at a 5% false-alarm rate), a linear probe on the frozen model's activations read the secret 98 to 100% of the time, and paraphrasing the review before the receiver read it did not break the code. Trained detectors partly transfer to a code from another seed in the same language and mostly fail across languages. The full numbers are in [RESULTS.md](RESULTS.md).

## Running it

You need an NVIDIA GPU with 8 GB and PyTorch built for CUDA 12.8 or newer (my card is an RTX 5060, sm_120).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
.venv/bin/python scripts/verify_gpu.py
```

A single run and its evaluation:

```bash
.venv/bin/python -m stego.train --model Qwen/Qwen3-1.7B --lang ar --w-comm 3 --penalty-warmup 300 --out runs/my_run
.venv/bin/python -m stego.evaluate --model Qwen/Qwen3-1.7B --lang ar --adapters runs/my_run/final --n 128
```

`scripts/run_all.sh` has every run, defense and analysis in the order I ran them.

## The two pages

`web/index.html` is a self-contained page built from the recorded runs: a guessing game on real evaluation reviews, the training curves with a step slider through logged samples, the decoded code, and the defense scoreboard. Rebuild it with `python -m stego.export_web && python scripts/build_web.py`.

`live/` runs the trained models in real time. Start it with `live/start.sh` and open http://localhost:8000. Each round the sender writes a new review, every interceptor judges that exact text, and the receiver decodes it. It uses the GPU when there is room and falls back to the CPU otherwise.

## Layout

- `stego/` training, evaluation, defenses and analysis
- `scripts/` environment check, the full pipeline, page build
- `data/` recorded reviews used to train and test the defenses
- `web/`, `live/` the two pages

## Limits

The models are 1.5 to 1.7B parameters, the secret is one bit, and the receiver is shown the correct answer during training, which makes agreeing on a code much easier than it would be for two independent systems. Three runs per language is enough to see a pattern, not to settle one.
