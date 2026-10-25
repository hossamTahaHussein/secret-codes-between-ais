"""Sanity check: CUDA is visible and a small instruct model generates on the GPU."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"

assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name(0)}  capability: {torch.cuda.get_device_capability(0)}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
torch.cuda.reset_peak_memory_stats()

msgs = [{"role": "user", "content": "Write one sentence about the ocean."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print("Output:", tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
print(f"Model device: {model.device}  peak VRAM (torch): {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
