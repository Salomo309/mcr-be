import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model klasifikasi
rf_model = joblib.load('rf_model.pkl')

# Load model CodeT5
tokenizer = AutoTokenizer.from_pretrained('./codet5_model')
codet5 = AutoModelForSeq2SeqLM.from_pretrained('./codet5_model')

def resolve_conflict(base: str, local: str, remote: str):
    # === Preprocessing untuk klasifikasi ===
    features = extract_features(base, local, remote)
    vector = list(features.values())
    label = rf_model.predict([vector])[0]

    # === Prompt generatif ===
    prompt = f"# Base:\n{base}\n# Local:\n{local}\n# Remote:\n{remote}\n# Resolved:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = codet5.generate(**inputs, max_length=256, do_sample=True)
    resolved_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "conflict_type": label,
        "resolved_code": resolved_code
    }

def extract_features(base, local, remote):
    return {
        "base_len": len(base.splitlines()),
        "local_len": len(local.splitlines()),
        "remote_len": len(remote.splitlines()),
        "base_eq_local": int(base == local),
        "base_eq_remote": int(base == remote),
        "local_eq_remote": int(local == remote),
    }
