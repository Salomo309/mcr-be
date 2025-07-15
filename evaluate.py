import json
import time
from sklearn.metrics import classification_report, accuracy_score
from resolve import resolve_conflict  # Pastikan resolve.py dan test.jsonl satu folder
from tqdm import tqdm
from collections import defaultdict, Counter

from sacrebleu.metrics import BLEU
from tree_sitter import Parser
from tree_sitter_languages import get_language
import re

# ====== Setup untuk CodeBLEU ======
def tokenize_code(code):
    return re.findall(r"[\w]+|[^\s\w]", code)

def compute_bleu(preds, refs):
    return BLEU().corpus_score(preds, [refs]).score

def weighted_ngram_match(preds, refs, keywords, n=4):
    total_score = 0
    for pred, ref in zip(preds, refs):
        pred_tokens = tokenize_code(pred)
        ref_tokens = tokenize_code(ref)
        pred_ngrams = Counter([" ".join(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        ref_ngrams = Counter([" ".join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        score = overlap / total if total > 0 else 0
        total_score += score
    return (total_score / len(preds)) * 100

def extract_syntax_nodes(code, parser):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    result = []
    def traverse(node):
        result.append(node.type)
        for child in node.children:
            traverse(child)
    traverse(root_node)
    return result

def is_trivial_code(code: str) -> bool:
    trivial_patterns = [
        r'^\s*$', r'^\s*pass\s*$', r'^\s*\{\s*\}\s*$',
        r'^\s*def .*:\s*pass\s*$', r'^\s*if .*:\s*pass\s*$',
        r'^\s*while .*:\s*pass\s*$',
    ]
    for pattern in trivial_patterns:
        if re.match(pattern, code.strip()):
            return True
    return False

def syntax_match(preds, refs, lang="python"):
    language = get_language(lang)
    parser = Parser()
    parser.set_language(language)
    total_score = 0
    valid_count = 0
    for pred, ref in zip(preds, refs):
        if is_trivial_code(pred): continue
        try:
            pred_nodes = extract_syntax_nodes(pred, parser)
            ref_nodes = extract_syntax_nodes(ref, parser)
            pred_counts = Counter(pred_nodes)
            ref_counts = Counter(ref_nodes)
            overlap = sum((pred_counts & ref_counts).values())
            total = sum(pred_counts.values())
            score = overlap / total if total > 0 else 0
            total_score += score
            valid_count += 1
        except:
            continue
    return (total_score / valid_count) * 100 if valid_count else 0

def compute_codebleu(preds, refs, keywords, lang="python", weights=(0.25, 0.25, 0.25, 0.25)):
    bleu = compute_bleu(preds, refs)
    weighted_match = weighted_ngram_match(preds, refs, keywords)
    syntax = syntax_match(preds, refs, lang=lang)
    dataflow = 0.0  # diabaikan

    score = (
        weights[0] * bleu +
        weights[1] * weighted_match +
        weights[2] * syntax +
        weights[3] * dataflow
    )

    return {
        "bleu": bleu,
        "weighted_ngram_match": weighted_match,
        "syntax_match": syntax,
        "codebleu": score
    }

# ====== Load Data Uji ======
TEST_PATH = "test.jsonl"
LIMIT = 5000
data = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= LIMIT: break
        data.append(json.loads(line.strip()))

# ====== Evaluasi Resolusi ======
y_true = []
y_pred = []
times = []
time_per_class = defaultdict(list)
count_per_class = Counter()

# Tambahan untuk CodeBLEU
kompleks_preds = []
kompleks_refs = []

print(f"Melakukan evaluasi pada {LIMIT} data uji...\n")
for sample in tqdm(data):
    input_text = sample["input"]
    label = sample["label"]
    try:
        base = input_text.split("### LOCAL")[0].replace("### BASE\n", "").strip()
        local = input_text.split("### LOCAL\n")[1].split("### REMOTE")[0].strip()
        remote = input_text.split("### REMOTE\n")[1].strip()
    except:
        continue

    start_time = time.time()
    result = resolve_conflict(base, local, remote)
    elapsed = time.time() - start_time

    pred_label = result["conflict_type"]
    y_true.append(label)
    y_pred.append(pred_label)
    times.append(elapsed)
    time_per_class[pred_label].append(elapsed)
    count_per_class[pred_label] += 1

    if pred_label == "Kompleks":
        kompleks_preds.append(result["resolved_code"])
        kompleks_refs.append(sample["resolved"])

# ====== Evaluasi Klasifikasi ======
print("\n=== HASIL PENGUJIAN KLASIFIKASI ===")
print(classification_report(y_true, y_pred, digits=4))
print(f"Akurasi: {accuracy_score(y_true, y_pred):.4f}")
print(f"\nRata-rata waktu penyelesaian per konflik: {sum(times)/len(times):.4f} detik")

print("\n=== Rata-rata waktu per jenis resolusi yang diprediksi ===")
for label in ["A", "B", "Kompleks"]:
    durations = time_per_class[label]
    if durations:
        avg_time = sum(durations) / len(durations)
        print(f"{label:8}: {avg_time:.4f} detik ({len(durations)} kasus)")
    else:
        print(f"{label:8}: Tidak ada prediksi")

print("\n=== Distribusi jumlah prediksi per jenis resolusi ===")
total = sum(count_per_class.values())
for label in ["A", "B", "Kompleks"]:
    count = count_per_class[label]
    pct = (count / total) * 100 if total > 0 else 0
    print(f"{label:8}: {count:4} kasus ({pct:.2f}%)")

# ====== Evaluasi CodeBLEU pada kasus Kompleks ======
if kompleks_preds:
    print("\n=== EVALUASI KODE UNTUK KONFLIK KOMPLEKS ===")
    result = compute_codebleu(kompleks_preds, kompleks_refs, keywords=[], lang="python")
    for k, v in result.items():
        print(f"{k:25}: {v:.4f}")
else:
    print("\nTidak ada prediksi 'Kompleks' untuk evaluasi CodeBLEU.")
