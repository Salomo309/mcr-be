import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
from peft import PeftModel, PeftConfig
from difflib import SequenceMatcher
from collections import Counter
import torch

# ====== PATH MODEL ======
RF_MODEL_PATH = "models/rf_model.pkl"
ARTIFACT_PATH = "artifact/data_advanced_3.pkl"
LLM_MODEL_PATH = "models/codet5_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Load Model Random Forest ======
with open(RF_MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

# ====== Load label encoder + info fitur ======
with open(ARTIFACT_PATH, "rb") as f:
    artifacts = pickle.load(f)
    label_encoder = artifacts["label_encoder"]

# ====== Load CodeT5 + PEFT ======
peft_config = PeftConfig.from_pretrained(LLM_MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
model_llm = PeftModel.from_pretrained(base_model, LLM_MODEL_PATH).to(device)
model_llm.eval()

# ====== Untuk fitur embedding ======
encoder_model = T5EncoderModel.from_pretrained(peft_config.base_model_name_or_path).to(device)
encoder_model.eval()

# ====== Feature Engineering ======
def extract_edit_sequence(base: str, other: str):
    seq = SequenceMatcher(None, base.split(), other.split())
    ops = []
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag == "equal":
            ops.extend(["="] * (i2 - i1))
        elif tag == "replace":
            ops.extend(["↔"] * max(i2 - i1, j2 - j1))
        elif tag == "insert":
            ops.extend(["+"] * (j2 - j1))
        elif tag == "delete":
            ops.extend(["-"] * (i2 - i1))
    return ops if ops else ["∅"]

def get_edit_bow(seq):
    ops = ["+", "-", "=", "↔", "∅"]
    count = Counter(seq)
    return [count.get(op, 0) for op in ops]

def extract_all_features(base, local, remote, tokenizer, encoder_model, device="cpu"):
    full_input = f"<base>\n{base}\n<local>\n{local}\n<remote>\n{remote}"
    inputs = tokenizer(full_input, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = encoder_model(**inputs)
        pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    ops_ao = extract_edit_sequence(base, local)
    ops_bo = extract_edit_sequence(base, remote)
    bow_ao = get_edit_bow(ops_ao)
    bow_bo = get_edit_bow(ops_bo)

    features = np.hstack([pooled_embedding, bow_ao + bow_bo])
    return features.reshape(1, -1)

# ====== Resolver utama (dipanggil Flask) ======
def resolve_conflict(base: str, local: str, remote: str):
    features = extract_all_features(base, local, remote, tokenizer, encoder_model, device)
    pred_label_idx = rf_model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_label_idx])[0]

    if pred_label == "A":
        return {
            "conflict_type": "A",
            "resolved_code": local.strip()
        }
    elif pred_label == "B":
        return {
            "conflict_type": "B",
            "resolved_code": remote.strip()
        }
    else:
        # Jika kompleks, pakai LLM generatif
        prompt = f"<base>\n{base}\n<local>\n{local}\n<remote>\n{remote}"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model_llm.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )

        resolved = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        resolved = resolved.replace("<end>", "").strip()
        return {
            "conflict_type": "Kompleks",
            "resolved_code": resolved.strip()
        }
