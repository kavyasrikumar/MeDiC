"""
Test SciSpacy Improvements for Higher F1
=========================================
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.sapbert_matcher import SapBERTMatcher


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def compare_predictions(predicted_anns, gold_anns, overlap_threshold=0.5):
    """Compare predictions vs gold standard."""
    tp, fp = 0, 0
    matched_gold = set()

    for pred in predicted_anns:
        pred_start = pred.get('start_offset', 0)
        pred_end = pred.get('end_offset', 0)
        pred_cui = pred.get('canonical_concept', '')

        matched = False
        for i, gold in enumerate(gold_anns):
            if i in matched_gold:
                continue

            gold_start = gold.get('start_offset', 0)
            gold_end = gold.get('end_offset', 0)
            gold_cui = gold.get('canonical_concept', '')

            if pred_cui != gold_cui:
                continue

            overlap_start = max(pred_start, gold_start)
            overlap_end = min(pred_end, gold_end)

            if overlap_end > overlap_start:
                overlap_len = overlap_end - overlap_start
                pred_len = pred_end - pred_start
                gold_len = gold_end - gold_start
                overlap_ratio = overlap_len / max(pred_len, gold_len)

                if overlap_ratio >= overlap_threshold:
                    tp += 1
                    matched_gold.add(i)
                    matched = True
                    break

        if not matched:
            fp += 1

    fn = len(gold_anns) - len(matched_gold)
    return tp, fp, fn


def evaluate_config(matcher, docs, threshold, gap):
    """Evaluate a specific configuration."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for doc in docs:
        predictions = matcher.predict_on_document(doc, threshold=threshold, min_confidence_gap=gap)
        gold_anns = doc.get('annotations', [])
        tp, fp, fn = compare_predictions(predictions, gold_anns)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return calculate_metrics(total_tp, total_fp, total_fn)


def main():
    print("=" * 80)
    print("TESTING IMPROVEMENTS FOR HIGHER F1")
    print("=" * 80)
    print()

    # Load matcher
    print("Loading SapBERT matcher...")
    matcher = SapBERTMatcher()
    matcher.load_embeddings("models/sapbert_embeddings.json")

    # Load dev set
    print("Loading development set...")
    with open("data/kaggle_splits/dev.json", 'r') as f:
        docs = json.load(f)[:50]  # Sample
    print(f"   Loaded {len(docs)} documents")
    print()

    # Test 1: Lower thresholds (more permissive)
    print("=" * 80)
    print("TEST 1: Lower Thresholds (trading precision for recall)")
    print("=" * 80)
    print()
    print(f"{'Config':>30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 90)

    configs = [
        # Current best
        {'t': 0.75, 'g': 0.05, 'label': 'Current (T=0.75, G=0.05)'},
        # Lower thresholds
        {'t': 0.70, 'g': 0.05, 'label': 'T=0.70, G=0.05'},
        {'t': 0.65, 'g': 0.05, 'label': 'T=0.65, G=0.05'},
        {'t': 0.60, 'g': 0.05, 'label': 'T=0.60, G=0.05'},
        # Lower gap (more permissive)
        {'t': 0.70, 'g': 0.02, 'label': 'T=0.70, G=0.02'},
        {'t': 0.65, 'g': 0.02, 'label': 'T=0.65, G=0.02'},
        # No gap (max recall)
        {'t': 0.70, 'g': 0.00, 'label': 'T=0.70, G=0.00'},
        {'t': 0.65, 'g': 0.00, 'label': 'T=0.65, G=0.00'},
        {'t': 0.60, 'g': 0.00, 'label': 'T=0.60, G=0.00'},
    ]

    best_f1 = 0
    best_config = None

    for cfg in configs:
        m = evaluate_config(matcher, docs, cfg['t'], cfg['g'])
        print(f"{cfg['label']:>30} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} "
              f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
        
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_config = cfg

    print()
    print(f"Best F1: {best_f1:.4f} with {best_config['label']}")
    print()

    # Analyze why recall is low
    print("=" * 80)
    print("DIAGNOSIS: Why is recall low?")
    print("=" * 80)
    print()

    # Count gold CUIs vs our embeddings
    gold_cuis = set()
    our_cuis = set(matcher.concept_embeddings.keys())
    
    for doc in docs:
        for ann in doc.get('annotations', []):
            gold_cuis.add(ann.get('canonical_concept', ''))

    missing_cuis = gold_cuis - our_cuis
    covered_cuis = gold_cuis & our_cuis

    print(f"Gold standard unique CUIs: {len(gold_cuis)}")
    print(f"Our embedding CUIs: {len(our_cuis)}")
    print(f"CUIs we CAN match: {len(covered_cuis)} ({100*len(covered_cuis)/len(gold_cuis):.1f}%)")
    print(f"CUIs we're MISSING: {len(missing_cuis)} ({100*len(missing_cuis)/len(gold_cuis):.1f}%)")
    print()

    if missing_cuis:
        print("Sample missing CUIs (first 10):")
        for cui in list(missing_cuis)[:10]:
            print(f"   {cui}")


if __name__ == "__main__":
    main()
