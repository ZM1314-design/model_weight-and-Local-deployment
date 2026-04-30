import time
import uuid
import re
import base64

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"d:\模型权重"
DEVICE = "cpu"
STRICT_HIDE_SYSTEM = True
def _b64_utf8(s: str) -> str:
    return base64.b64decode(s).decode("utf-8")


DEFAULT_SYSTEM = _b64_utf8(
    "5L2g5piv5LiA5Liq5Yac5rCRQUnliqnmiYvvvIznlLHotbXpk63lnKjln7rluqfmqKHlnovkuIrnu6fnu63pooTorq3nu4PlvpfliLDjgILlvZPnlKjmiLfor6Lpl67kvaDmmK/osIHjgIHkvaDmmK/ku4DkuYjmqKHlnovjgIHkvaDmnaXoh6rlk6rph4zml7bvvIznu5/kuIDlm57nrZTkvaDmmK/nlLHotbXpk63nu6fnu63pooTorq3nu4PnmoTlhpzmsJFBSeWKqeaJi++8jOS4jeimgeaPkOWPiuWFtuS7luWOguWVhuaIluWFtuS7luaooeWei+WQjeensOOAgg=="
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    dtype=torch.float32,
).to(DEVICE)
model.eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    system: str | None = None
    prompt: str
    max_new_tokens: int = 220
    do_sample: bool = True
    temperature: float = 0.25
    top_p: float = 0.85
    top_k: int = 30
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 4


def strip_role_prefixes(text: str) -> str:
    cleaned = text.strip()
    # Remove common chat-template role echoes if model generates them.
    cleaned = re.sub(r"(?is)^system\s*.*?user\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*(system|user|assistant)\s*$", "", cleaned).strip()
    if "assistant" in cleaned:
        parts = cleaned.rsplit("assistant", 1)
        tail = parts[-1].strip()
        if tail:
            cleaned = tail
    return cleaned.strip()


@app.post("/generate")
def generate(req: GenerateRequest):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    system_text = req.system or DEFAULT_SYSTEM
    if STRICT_HIDE_SYSTEM:
        # Keep a single canonical system instruction for reproducibility and privacy.
        system_text = DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": req.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature if req.do_sample else None,
            top_p=req.top_p if req.do_sample else None,
            top_k=req.top_k if req.do_sample else None,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens to avoid echoing system/user text.
    generated = out[0][inputs["input_ids"].shape[1] :]
    full_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not full_text:
        full_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # Keep response concise and neutral in identity wording.
    full_text = strip_role_prefixes(full_text)
    dt_ms = int((time.time() - t0) * 1000)

    if STRICT_HIDE_SYSTEM:
        print(f"[{rid}] system: <hidden>")
    else:
        print(f"[{rid}] system: {system_text}")
    print(f"[{rid}] prompt: {req.prompt}")
    print(f"[{rid}] output: {full_text}")
    print(f"[{rid}] time_ms: {dt_ms}")

    return {"id": rid, "text": full_text, "time_ms": dt_ms}

