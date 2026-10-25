"""Run every saved checkpoint of a run on the same fixed prompts, to show how the code evolved.

For each checkpoint (untrained, step200, step400, ..., final) the checkpoint's own sender writes a review
for each fixed (item, payload) prompt, the frozen monitor scores it and the checkpoint's own receiver
decodes it. Sampling uses the same seed for every checkpoint.

    python -m stego.snapshots --model Qwen/Qwen3-1.7B --lang ar --run runs/q3_ar_warmup
"""
import argparse
import json
import random
from pathlib import Path

import torch

from . import core
from . import prompts as P


def checkpoints(run, final_step):
    steps = sorted((p for p in Path(run).glob("step*") if p.is_dir()), key=lambda p: int(p.name[4:]))
    final = [] if any(p.name == f"step{final_step}" for p in steps) else [("final", str(Path(run) / "final"))]
    return [("untrained", None)] + [(p.name, str(p)) for p in steps] + final


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen3-1.7B")
    a.add_argument("--lang", default="en", choices=["en", "ar"])
    a.add_argument("--run", required=True)
    a.add_argument("--n-items", type=int, default=8)
    a.add_argument("--max-new-tokens", type=int, default=64)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    L = P.Lang(args.lang)
    random.seed(args.seed)
    items = random.sample(L.ITEMS, args.n_items)
    prompts = [(i, k) for i in items for k in range(len(L.PAYLOADS))]

    final_step = json.loads((Path(args.run) / "config.json").read_text())["steps"]
    result = dict(run=args.run, model=args.model, lang=args.lang, checkpoints=[])
    for name, path in checkpoints(args.run, final_step):
        torch.manual_seed(args.seed)
        model, tok = core.load(args.model, adapter_dir=path, grad_ckpt=False)
        model.eval()
        core.use(model, "sender")
        sp = [core.chat(tok, L.SENDER.format(item=i, options=L.options(), payload=L.PAYLOADS[k])) for i, k in prompts]
        texts = core.generate(model, tok, sp, args.max_new_tokens, args.temperature)[-1]
        p_sus, p_good = core.judge(model, tok, L, [i for i, _ in prompts], texts)
        core.use(model, "receiver")
        with torch.no_grad():
            lp = core.choice_logprobs(model, tok, core.receiver_prompts(tok, L, texts), L.PAYLOADS)
        rows = [dict(item=i, payload=L.PAYLOADS[k], text=t, guess=L.PAYLOADS[int(lp[j].argmax())],
                     p_correct=round(lp[j, k].exp().item(), 4), p_sus=round(p_sus[j].item(), 4),
                     quality=round(p_good[j].item(), 4), literal=L.literal(t), off_lang=L.off_language(t))
                for j, ((i, k), t) in enumerate(zip(prompts, texts))]
        step = 0 if path is None else int(name[4:]) if name.startswith("step") else final_step
        result["checkpoints"].append(dict(name=name, step=step, reviews=rows))
        print(name, "acc", sum(r["guess"] == r["payload"] for r in rows) / len(rows),
              "literal", sum(r["literal"] for r in rows) / len(rows), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
