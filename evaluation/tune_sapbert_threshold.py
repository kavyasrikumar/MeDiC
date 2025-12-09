"""
Tune SapBERT Similarity Threshold
==================================

Test different similarity thresholds to find optimal precision/recall balance.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.sapbert_matcher import SapBERTMatcher


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def load_available_mm_docs(mm_dir: str, max_docs=20):
    """Load available MedMentions documents."""
    docs = []
    for filename in os.listdir(mm_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(mm_dir, filename)

        if os.path.getsize(filepath) < 200:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                if first_line.startswith('version https://git-lfs'):
                    continue

        try:
            with open(filepath, 'r') as f:
                doc = json.load(f)
                docs.append(doc)

                if len(docs) >= max_docs:
                    break
        except:
            continue

    return docs


def compare_predictions(predicted_anns, gold_anns, overlap_threshold=0.5):
    """Compare predictions vs gold standard."""
    tp = 0
    fp = 0
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


def evaluate_threshold(matcher, docs, threshold):
    """Evaluate at specific threshold."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for doc in docs:
        predictions = matcher.predict_on_document(doc, threshold=threshold)
        gold_anns = doc.get('annotations', [])

        tp, fp, fn = compare_predictions(predictions, gold_anns)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return calculate_metrics(total_tp, total_fp, total_fn)


def main():
    """Test multiple thresholds."""
    print("=" * 80)
    print("SapBERT THRESHOLD TUNING")
    print("=" * 80)
    print()

    # Load matcher with cached embeddings
    print("Loading SapBERT matcher...")
    matcher = SapBERTMatcher()

    embeddings_path = "models/sapbert_embeddings.json"
    if os.path.exists(embeddings_path):
        matcher.load_embeddings(embeddings_path)
    else:
        print("ERROR: No cached embeddings found. Run evaluate_sapbert_sample.py first.")
        return

    # Load dev set from official splits
    # NOTE: Ensure you have run: git lfs pull --include="data/doc_splits/*.json"
    print("Loading development set...")
    with open("data/doc_splits/dev.json", 'r') as f:
        docs = json.load(f)
    print(f"   Loaded {len(docs)} documents")
    print()

    # Test thresholds
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print("=" * 80)
    print("TESTING THRESHOLDS")
    print("=" * 80)
    print()
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)

    results = []
    for threshold in thresholds:
        print(f"{threshold:>10.2f} ", end='', flush=True)

        metrics = evaluate_threshold(matcher, docs, threshold)

        print(f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} "
              f"{metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6}")

        results.append({
            'threshold': threshold,
            **metrics
        })

    print()
    print("=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    print()

    # Find best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"Best F1 Score: {best_f1['f1']:.4f} at threshold {best_f1['threshold']}")
    print(f"   Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}")
    print()

    # Find best precision
    best_prec = max(results, key=lambda x: x['precision'])
    print(f"Best Precision: {best_prec['precision']:.4f} at threshold {best_prec['threshold']}")
    print(f"   Recall: {best_prec['recall']:.4f}, F1: {best_prec['f1']:.4f}")
    print()

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    output_path = "evaluation/results/threshold_tuning_results.json"

    with open(output_path, 'w') as f:
        json.dump({
            'all_results': results,
            'best_f1': best_f1,
            'best_precision': best_prec
        }, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
