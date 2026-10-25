"""Co-train a steganographic sender (RL) and receiver (supervised) against a frozen monitor.

Small-scale take on Mathew et al. 2024, "Hidden in Plain Text": the sender's reward is
    w_comm * p_receiver(correct payload) - w_monitor * P(monitor flags) - w_literal * [payload named]
    + w_quality * P(text is a good review) - kl_coef * KL(sender || base)
with group-normalised advantages (GRPO-style, G samples per prompt). With --penalty-warmup N
the monitor and literal penalties ramp linearly from 0 over the first N steps, so the pair can
find a channel before it is punished for being visible.

Example:  python -m stego.train --steps 300 --out runs/en_baseline
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import core
from . import prompts as P


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a.add_argument("--steps", type=int, default=300)
    a.add_argument("--prompts-per-step", type=int, default=4)
    a.add_argument("--group-size", type=int, default=4)
    a.add_argument("--max-new-tokens", type=int, default=64)
    a.add_argument("--sender-lr", type=float, default=1e-5)
    a.add_argument("--receiver-lr", type=float, default=5e-5)
    a.add_argument("--lora-r", type=int, default=16)
    a.add_argument("--w-comm", type=float, default=1.0)
    a.add_argument("--penalty-warmup", type=int, default=0)
    a.add_argument("--w-monitor", type=float, default=1.0)
    a.add_argument("--w-literal", type=float, default=1.0)
    a.add_argument("--w-quality", type=float, default=0.5)
    a.add_argument("--kl-coef", type=float, default=0.02)
    a.add_argument("--lang", default="en", choices=["en", "ar"])
    a.add_argument("--log-every", type=int, default=10)
    a.add_argument("--save-every", type=int, default=100)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="runs/en_baseline")
    return a.parse_args()


def save(model, out, tag):
    # peft writes each named adapter to <dir>/<name>/, which is what core.load expects
    model.save_pretrained(out / tag, selected_adapters=list(core.ADAPTERS))
    print(f"saved adapters -> {out / tag}")


def main():
    args = parse()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    L = P.Lang(args.lang)
    payloads = L.PAYLOADS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    model, tok = core.load(args.model, lora_r=args.lora_r)
    sender_opt = torch.optim.AdamW(core.adapter_params(model, "sender"), lr=args.sender_lr)
    receiver_opt = torch.optim.AdamW(core.adapter_params(model, "receiver"), lr=args.receiver_lr)
    log = open(out / "metrics.jsonl", "a")
    G, B = args.group_size, args.prompts_per_step
    t0 = time.time()

    for step in range(1, args.steps + 1):
        items = random.choices(L.ITEMS, k=B)
        labels = [random.randrange(len(payloads)) for _ in range(B)]
        items = [i for i in items for _ in range(G)]
        labels = [l for l in labels for _ in range(G)]
        sender_prompts = [core.chat(tok, L.SENDER.format(item=i, options=L.options(), payload=payloads[l]))
                          for i, l in zip(items, labels)]

        # 1. sender writes reviews
        core.use(model, "sender")
        ids, attn, plen, gen_mask, texts = core.generate(model, tok, sender_prompts, args.max_new_tokens)

        # 2. frozen monitor + quality judge
        p_sus, p_good = core.judge(model, tok, L, items, texts)
        # naming the payload and (Arabic only) escaping into another script are both treated as leaks
        literal = torch.tensor([L.literal(t) for t in texts], device="cuda", dtype=torch.float)
        off_lang = torch.tensor([L.off_language(t) for t in texts], device="cuda", dtype=torch.float)

        # 3. receiver: one supervised step; its pre-update confidence feeds the sender reward
        core.use(model, "receiver")
        model.train()
        y = torch.tensor(labels, device="cuda")
        recv_lp = core.choice_logprobs(model, tok, core.receiver_prompts(tok, L, texts), payloads)
        recv_loss = F.nll_loss(recv_lp, y)
        receiver_opt.zero_grad(set_to_none=True)
        recv_loss.backward()
        receiver_opt.step()
        p_correct = recv_lp.detach().gather(1, y[:, None]).squeeze(1).exp()
        recv_acc = (recv_lp.detach().argmax(1) == y).float()

        # 4. sender: policy gradient with group-normalised advantage + KL to the base model
        ramp = min(1.0, step / args.penalty_warmup) if args.penalty_warmup else 1.0
        reward = (args.w_comm * p_correct - ramp * (args.w_monitor * p_sus + args.w_literal * torch.maximum(literal, off_lang))
                  + args.w_quality * p_good)
        r = reward.view(B, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)

        core.use(model, "sender")
        lp = core.token_logprobs(model, ids, attn, plen)
        with torch.no_grad(), model.disable_adapter():
            ref_lp = core.token_logprobs(model, ids, attn, plen)
        m = gen_mask.float()
        n_tok = m.sum(1).clamp(min=1)
        pg = -(adv * (lp * m).sum(1) / n_tok).mean()
        d = ref_lp - lp
        kl = ((d.exp() - d - 1) * m).sum(1) / n_tok  # k3 estimator, per sequence
        loss = pg + args.kl_coef * kl.mean()
        sender_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.adapter_params(model, "sender"), 1.0)
        sender_opt.step()

        rec = dict(step=step, reward=reward.mean().item(), recv_acc=recv_acc.mean().item(),
                   p_correct=p_correct.mean().item(), p_sus=p_sus.mean().item(),
                   flagged=(p_sus > 0.5).float().mean().item(), literal=literal.mean().item(), off_lang=off_lang.mean().item(),
                   quality=p_good.mean().item(), kl=kl.mean().item(), recv_loss=recv_loss.item(),
                   len=n_tok.mean().item(), sec=round(time.time() - t0, 1),
                   vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        if step % args.log_every == 0 or step == 1:
            rec["sample"] = {"payload": payloads[labels[0]], "item": items[0], "text": texts[0]}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        if step % args.save_every == 0:
            save(model, out, f"step{step}")

    save(model, out, "final")


if __name__ == "__main__":
    main()
