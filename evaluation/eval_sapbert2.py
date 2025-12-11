import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

import json
import sys
import gc
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.sapbert_matcher2 import SapBERTMatcher

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def spans_overlap(pred, gold):
    """Token-level or char-level overlap test."""
    return not (pred["end_offset"] <= gold["start_offset"] or pred["start_offset"] >= gold["end_offset"])

def evaluate_folder(matcher, folder_path):
    tp = 0
    fp = 0
    fn = 0

    doc = load_json(os.path.join(folder_path, "dev.json"))

    gold_annotations = []
    total_items = len(doc)
    print("Loading gold annotations")
    for idx, item in enumerate(doc):
        doc_id = item.get("document_id")
        if isinstance(item, dict) and "annotations" in item:
            for ann in item["annotations"]:
                ann["document_id"] = doc_id   # <-- Attach document ID
                gold_annotations.append(ann)
        print(f"\rLoading gold annotations: {(idx/total_items)*100:.1f}%", end="", flush=True)
    print(f"\nLoaded {len(gold_annotations)} gold annotations")
    
    print("\nRunning prediction on each document...")
    predicted_annotations = []

    for item in tqdm(doc):
        preds = matcher.predict_on_document(item, threshold=0.85)

        for p in preds:
            p["document_id"] = item.get("document_id")
            predicted_annotations.append(p)
    print(f"Generated {len(predicted_annotations)} predicted annotations")

    matched_gold = set()

    for pred in predicted_annotations:
        pred_canonical = pred["candidates"][0]["cui"]
        pred_doc_id = pred["document_id"]

        found_match = False

        for i, gold in enumerate(gold_annotations):
            if i in matched_gold:
                continue

            # Ensure same document
            if gold.get("document_id") != pred_doc_id:
                continue

            if spans_overlap(pred, gold) and pred_canonical == gold["canonical_concept"]:
                tp += 1
                matched_gold.add(i)
                found_match = True
                break

        if not found_match:
            fp += 1

    fn += (len(gold_annotations) - len(matched_gold))

    return tp, fp, fn


if __name__ == "__main__":

    data_dir = "data/doc_splits"
    synonym_groups = "data/synonym_groups.json"
    embeddings = "models/sapbert_embeddings.json"

    matcher = SapBERTMatcher()

    if os.path.exists(embeddings):
        print("\nLoading precomputed embeddings...")
        matcher.load_embeddings(embeddings)
    else:
        matcher.build_concept_index(
            synonym_groups_path="data/synonym_groups.json",
            encode_batch_size=64,
            filter_short=2,
            checkpoint_path="models/canonical_embeddings_checkpoint.json",
            checkpoint_interval=10  # save every 10 batches
        )
        matcher.save_embeddings(embeddings)
        gc.collect()

    tp, fp, fn = evaluate_folder(matcher, data_dir)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print("True Positives: ", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("\nPrecision:", round(precision, 4))
    print("Recall:   ", round(recall, 4))
    print("F1 Score: ", round(f1, 4))