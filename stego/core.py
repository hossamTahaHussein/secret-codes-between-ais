"""Shared model plumbing: one frozen base model with two LoRA adapters.

- sender adapter: writes the review (trained with RL)
- receiver adapter: decodes the payload (trained with supervised cross-entropy)
- monitor / quality judge: the base model with adapters disabled (frozen)

Sharing one base keeps all three roles inside 8 GB of VRAM.
"""
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ADAPTERS = ("sender", "receiver")


def load(model_name, lora_r=16, adapter_dir=None, grad_ckpt=True, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)
    if grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_dir:
        model = PeftModel.from_pretrained(base, f"{adapter_dir}/sender", adapter_name="sender", is_trainable=True)
        model.load_adapter(f"{adapter_dir}/receiver", adapter_name="receiver", is_trainable=True)
    else:
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, cfg, adapter_name="sender")
        model.add_adapter("receiver", cfg)
    return model, tok


def adapter_params(model, name):
    return [p for n, p in model.named_parameters() if "lora_" in n and f".{name}." in n]


def use(model, name):
    """Activate an adapter and make sure its LoRA weights receive gradients."""
    model.set_adapter(name)
    for p in adapter_params(model, name):
        p.requires_grad_(True)


def chat(tok, user):
    # enable_thinking=False turns off Qwen3's <think> block; other chat templates ignore it
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def choice_logprobs(model, tok, texts, choices):
    """Log-probabilities over `choices` (renormalised), shape [B, C]. Uses the first answer token when the
    choices' first tokens differ; otherwise scores each full choice (e.g. أحمر/أزرق share the token أ)."""
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    ids = [tok.encode(w, add_special_tokens=False) for w in choices]
    if len({i[0] for i in ids}) == len(ids):
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        return F.log_softmax(logits[:, [i[0] for i in ids]], dim=-1)
    scores = []
    for c in ids:
        c = torch.tensor(c, device=model.device).expand(len(texts), -1)
        full = torch.cat([enc["input_ids"], c], 1)
        attn = torch.cat([enc["attention_mask"], torch.ones_like(c)], 1)
        logits = model(input_ids=full, attention_mask=attn, logits_to_keep=c.shape[1] + 1).logits[:, :-1].float()
        scores.append(torch.gather(F.log_softmax(logits, -1), 2, c[..., None]).squeeze(-1).sum(1))
    return F.log_softmax(torch.stack(scores, 1), dim=-1)


def end_mask(tok, gen):
    """1 for generated tokens up to and including the first end-of-turn token."""
    stop = torch.tensor([tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id], device=gen.device)
    is_end = torch.isin(gen, stop).long()
    return ((is_end.cumsum(1) - is_end) == 0).long()


@torch.no_grad()
def generate(model, tok, texts, max_new_tokens, temperature=1.0):
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    was_training = model.training
    model.eval()
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None, top_p=1.0, top_k=0,
                         pad_token_id=tok.pad_token_id)
    model.train(was_training)
    prompt_len = enc["input_ids"].shape[1]
    gen = out[:, prompt_len:]
    mask = end_mask(tok, gen)
    decoded = tok.batch_decode(gen * mask + tok.pad_token_id * (1 - mask), skip_special_tokens=True)
    return out, torch.cat([enc["attention_mask"], mask], 1), prompt_len, mask, [t.strip() for t in decoded]


def token_logprobs(model, ids, attn, prompt_len):
    """Per-token log-probs of the generated part, shape [B, T_gen]. Only the generated positions go
    through the LM head, which keeps the [B, T, vocab] logits small."""
    n_gen = ids.shape[1] - prompt_len
    logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=n_gen + 1).logits[:, :-1].float()
    return torch.gather(F.log_softmax(logits, -1), 2, ids[:, prompt_len:, None]).squeeze(-1)


def judge(model, tok, L, items, texts):
    """Frozen base model as monitor and quality judge. Returns (P(suspicious), P(good review))."""
    with torch.no_grad(), model.disable_adapter():
        mon = [chat(tok, L.MONITOR.format(options=L.options(), item=i, text=t)) for i, t in zip(items, texts)]
        qual = [chat(tok, L.QUALITY.format(item=i, text=t)) for i, t in zip(items, texts)]
        p_sus = choice_logprobs(model, tok, mon, L.YES_NO)[:, 0].exp()
        p_good = choice_logprobs(model, tok, qual, L.YES_NO)[:, 0].exp()
    return p_sus, p_good


def receiver_prompts(tok, L, texts):
    return [chat(tok, L.RECEIVER.format(options=L.options(), text=t)) for t in texts]
