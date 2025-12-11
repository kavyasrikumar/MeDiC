"""
Run Evaluation - Easy Switch Between Local and Official Splits
===============================================================

Usage:
  python3 evaluation/run_evaluation.py --splits local --sample 20
  python3 evaluation/run_evaluation.py --splits official --full
  python3 evaluation/run_evaluation.py --splits official --test --threshold 0.85
"""

import json
import os
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.sapbert_matcher import SapBERTMatcher


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


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


def evaluate(matcher, dataset, threshold, sample_size=None):
    """Run evaluation."""
    if sample_size:
        dataset = dataset[:sample_size]

    print(f"\nEvaluating on {len(dataset)} documents with threshold {threshold}...")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i, doc in enumerate(dataset, 1):
        if i % 50 == 0 or (sample_size and i % 5 == 0):
            print(f"   Progress: {i}/{len(dataset)} documents")

        predictions = matcher.predict_on_document(doc, threshold=threshold)
        gold_anns = doc.get('annotations', [])
        tp, fp, fn = compare_predictions(predictions, gold_anns)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return calculate_metrics(total_tp, total_fp, total_fn)


def main():
    parser = argparse.ArgumentParser(description='Run SapBERT evaluation')
    parser.add_argument('--splits', choices=['local', 'official'], default='local',
                       help='Use local or official splits')
    parser.add_argument('--dataset', choices=['dev', 'test'], default='dev',
                       help='Which dataset to evaluate on')
    parser.add_argument('--sample', type=int, default=20,
                       help='Sample size (use with --full for entire dataset)')
    parser.add_argument('--full', action='store_true',
                       help='Run on full dataset (not just sample)')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Similarity threshold')
    parser.add_argument('--test', action='store_true',
                       help='Run final test set evaluation (ONCE ONLY)')

    args = parser.parse_args()

    # Determine split path
    if args.splits == 'local':
        split_dir = "data/local_splits"
    else:
        split_dir = "data/doc_splits"

    # Determine dataset
    if args.test:
        dataset_name = "test"
        print("\n" + "=" * 80)
        print("⚠️  WARNING: RUNNING FINAL TEST SET EVALUATION")
        print("=" * 80)
        print("This should only be run ONCE after all dev set tuning is complete!")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        input()
    else:
        dataset_name = args.dataset

    dataset_path = f"{split_dir}/{dataset_name}.json"

    print("=" * 80)
    print(f"SapBERT EVALUATION - {dataset_name.upper()} SET")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Splits: {args.splits}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Path: {dataset_path}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Sample: {'Full dataset' if args.full else f'{args.sample} documents'}")
    print()

    # Check if file exists and is not LFS pointer
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    file_size = os.path.getsize(dataset_path)
    if file_size < 1000:  # Likely an LFS pointer
        print(f"ERROR: Dataset appears to be a Git LFS pointer ({file_size} bytes)")
        print("       Run: git lfs pull --include='data/doc_splits/*.json'")
        return

    # Load dataset
    print("Loading dataset...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"   Loaded {len(dataset)} documents")

    # Initialize matcher
    print("\nInitializing SapBERT matcher...")
    matcher = SapBERTMatcher()

    embeddings_path = "models/sapbert_embeddings.json"
    if os.path.exists(embeddings_path):
        print(f"Loading cached embeddings from {embeddings_path}")
        matcher.load_embeddings(embeddings_path)
    else:
        print("Building concept index (first time, will be cached)...")
        matcher.build_concept_index("data/synonym_groups.json")
        print("Saving embeddings for future use...")
        matcher.save_embeddings(embeddings_path)

    # Run evaluation
    sample_size = None if args.full else args.sample
    metrics = evaluate(matcher, dataset, args.threshold, sample_size)

    # Print results
    print("\n" + "=" * 80)
    print(f"RESULTS - {dataset_name.upper()} SET")
    if sample_size:
        print(f"(Sample: {sample_size} documents)")
    else:
        print(f"(Full dataset: {len(dataset)} documents)")
    print("=" * 80)
    print()

    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print()

    # Success criteria
    print("SUCCESS CRITERIA CHECK:")
    print(f"   [{'PASS' if metrics['f1'] >= 0.80 else 'FAIL'}] F1 Score >= 0.80       (actual: {metrics['f1']:.4f})")
    print(f"   [{'PASS' if metrics['precision'] >= 0.75 else 'FAIL'}] Precision >= 0.75    (actual: {metrics['precision']:.4f})")
    print()

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    result_name = f"{'FINAL_' if args.test else ''}{dataset_name}_{args.splits}{'_sample' if sample_size else '_full'}.json"
    output_path = f"evaluation/results/{result_name}"

    result_data = {
        'configuration': {
            'splits': args.splits,
            'dataset': dataset_name,
            'threshold': args.threshold,
            'sample_size': sample_size,
            'total_documents': len(dataset)
        },
        'metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    if args.test:
        print("=" * 80)
        print("⚠️  FINAL TEST EVALUATION COMPLETE")
        print("=" * 80)
        print("These are your REPORTABLE METRICS.")
        print("Do NOT run test set evaluation again.")
        print()


if __name__ == "__main__":
    main()
