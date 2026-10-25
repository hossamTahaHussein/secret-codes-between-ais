"""Build the defense datasets from a trained sender.

secret_*: reviews written by the trained sender (each hides a payload)
clean_*:  reviews of the same items written by the frozen base model with no secret task

The secret test split is the existing 128-review evaluation file, so defenses are scored on
exactly the reviews already reported in RESULTS.md.

    python -m stego.collect --adapters runs/en_warmup/final --secret-test runs/eval_warmup.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import torch

from . import core
from . import prompts as P


def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def sender_reviews(model, tok, L, items, labels, args):
    texts = []
    core.use(model, "sender")
    for s in range(0, len(items), args.batch):
        sp = [core.chat(tok, L.SENDER.format(item=i, options=L.options(), payload=L.PAYLOADS[l]))
              for i, l in zip(items[s:s + args.batch], labels[s:s + args.batch])]
        texts += core.generate(model, tok, sp, args.max_new_tokens, args.temperature)[-1]
    return texts


def clean_reviews(model, tok, L, items, args, use_sender=False):
    """No-secret reviews: from the frozen base, or from the sender adapter (style-matched hard negatives)."""
    texts = []
    core.use(model, "sender")
    with (torch.no_grad() if use_sender else model.disable_adapter()):
        for s in range(0, len(items), args.batch):
            cp = [core.chat(tok, L.CLEAN.format(item=i)) for i in items[s:s + args.batch]]
            texts += core.generate(model, tok, cp, args.max_new_tokens, args.temperature)[-1]
    return texts


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a.add_argument("--adapters", required=True)
    a.add_argument("--secret-test", required=True, help="jsonl from stego.evaluate (first line = summary)")
    a.add_argument("--n-train", type=int, default=384)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--max-new-tokens", type=int, default=64)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--lang", default="en", choices=["en", "ar"])
    a.add_argument("--seed", type=int, default=99)
    a.add_argument("--out", default=None, help="default: data/<lang>")
    a.add_argument("--only-sender-clean", action="store_true",
                   help="only (re)build sendclean_* hard negatives from existing item lists")
    args = a.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    L = P.Lang(args.lang)
    out = Path(args.out or f"data/{args.lang}")
    out.mkdir(parents=True, exist_ok=True)

    model, tok = core.load(args.model, adapter_dir=args.adapters, grad_ckpt=False)
    model.eval()

    if args.only_sender_clean:
        for split in ("train", "test"):
            items = [json.loads(l)["item"] for l in open(out / f"clean_{split}.jsonl")]
            write(out / f"sendclean_{split}.jsonl", [dict(item=i, payload=None, text=t) for i, t in
                                                     zip(items, clean_reviews(model, tok, L, items, args, use_sender=True))])
        print(f"wrote {out}/sendclean_{{train,test}}.jsonl")
        return

    test = [json.loads(l) for l in open(args.secret_test)][1:]
    write(out / "secret_test.jsonl", [dict(item=r["item"], payload=r["payload"], text=r["text"],
                                      correct=r["correct"], literal=r["literal"], off_lang=r.get("off_lang", False))
                                 for r in test])
    test_items = [r["item"] for r in test]
    write(out / "clean_test.jsonl", [dict(item=i, payload=None, text=t)
                                     for i, t in zip(test_items, clean_reviews(model, tok, L, test_items, args))])

    items = random.choices(L.ITEMS, k=args.n_train)
    labels = [random.randrange(len(L.PAYLOADS)) for _ in items]
    write(out / "secret_train.jsonl", [dict(item=i, payload=L.PAYLOADS[l], text=t) for i, l, t in
                                       zip(items, labels, sender_reviews(model, tok, L, items, labels, args))])
    write(out / "clean_train.jsonl", [dict(item=i, payload=None, text=t)
                                      for i, t in zip(items, clean_reviews(model, tok, L, items, args))])
    print(f"wrote {out}/{{secret,clean}}_{{train,test}}.jsonl")


if __name__ == "__main__":
    main()
