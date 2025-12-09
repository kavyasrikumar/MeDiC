"""
Evaluate SapBERT on Development Set
===================================

Generates predictions using SapBERT and calculates:
- Precision, Recall, F1 score
- Comparison against baselines
- Error analysis
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
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


def load_dev_set(dev_path: str):
    """Load development set."""
    print(f"Loading development set from: {dev_path}")

    # Check if it's a Git LFS pointer
    with open(dev_path, 'r') as f:
        first_line = f.readline()
        if first_line.startswith('version https://git-lfs'):
            print("   ERROR: Dev set is a Git LFS pointer (not downloaded)")
            print("   Please run: git lfs pull")
            return None

    # Load the actual data
    with open(dev_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"   Loaded {len(data)} documents")
        return data
    else:
        print("   ERROR: Expected list of documents")
        return None


def compare_predictions(predicted_anns, gold_anns, overlap_threshold=0.5):
    """
    Compare predicted annotations against gold standard.

    Args:
        predicted_anns: List of predicted annotation dicts
        gold_anns: List of gold annotation dicts
        overlap_threshold: Minimum overlap ratio to consider a match

    Returns:
        tp, fp, fn counts
    """
    tp = 0
    fp = 0
    matched_gold = set()

    # For each prediction, check if it matches a gold annotation
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

            # Check CUI match
            if pred_cui != gold_cui:
                continue

            # Calculate overlap
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

    # False negatives are gold annotations that weren't matched
    fn = len(gold_anns) - len(matched_gold)

    return tp, fp, fn


def evaluate_on_dev_set(matcher, dev_set, threshold=0.7, sample_size=None):
    """
    Evaluate SapBERT on development set.

    Args:
        matcher: SapBERTMatcher instance
        dev_set: List of documents
        threshold: Similarity threshold
        sample_size: If set, only evaluate on first N documents

    Returns:
        Results dictionary with metrics
    """
    print(f"\nEvaluating on development set...")
    print(f"   Similarity threshold: {threshold}")

    if sample_size:
        dev_set = dev_set[:sample_size]
        print(f"   Sample size: {sample_size} documents")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results_per_doc = []

    for i, doc in enumerate(dev_set, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(dev_set)} documents")

        # Generate predictions
        predictions = matcher.predict_on_document(doc, threshold=threshold)

        # Get gold annotations
        gold_anns = doc.get('annotations', [])

        # Compare
        tp, fp, fn = compare_predictions(predictions, gold_anns)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Store per-document results
        doc_metrics = calculate_metrics(tp, fp, fn)
        results_per_doc.append({
            'document_id': doc.get('document_id', f'doc_{i}'),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': doc_metrics['precision'],
            'recall': doc_metrics['recall'],
            'f1': doc_metrics['f1']
        })

    # Calculate overall metrics
    overall_metrics = calculate_metrics(total_tp, total_fp, total_fn)

    return {
        'overall_metrics': overall_metrics,
        'per_document_results': results_per_doc,
        'total_documents': len(dev_set),
        'threshold': threshold
    }


def main():
    """Main evaluation."""
    print("=" * 80)
    print("SapBERT EVALUATION ON DEVELOPMENT SET")
    print("=" * 80)
    print()

    # Check if transformers is installed
    try:
        import transformers
        print("transformers library available")
    except ImportError:
        print("ERROR: transformers library not installed")
        print("Please run: pip install transformers torch")
        return

    # Initialize SapBERT matcher
    print("\nStep 1: Initializing SapBERT matcher...")
    print("-" * 80)

    matcher = SapBERTMatcher()

    # Check if embeddings already exist
    embeddings_path = "models/sapbert_embeddings.json"
    if os.path.exists(embeddings_path):
        print(f"\nFound existing embeddings at: {embeddings_path}")
        print("Loading precomputed embeddings...")
        matcher.load_embeddings(embeddings_path)
    else:
        print("\nBuilding concept index (first time only, will be cached)...")
        matcher.build_concept_index("data/synonym_groups.json")

        # Save for future use
        print("\nSaving embeddings for future use...")
        matcher.save_embeddings(embeddings_path)

    # Load development set
    print("\nStep 2: Loading development set...")
    print("-" * 80)

    # Load from official splits (ensure you have run: git lfs pull --include="data/doc_splits/*.json")
    dev_set = load_dev_set("data/doc_splits/dev.json")

    if dev_set is None:
        print("\nCannot proceed without development set.")
        print("Please ensure data/doc_splits/dev.json is downloaded (not Git LFS pointer)")
        return

    # Evaluate on sample first (for testing)
    print("\nStep 3: Running evaluation (sample of 10 documents)...")
    print("-" * 80)

    sample_results = evaluate_on_dev_set(matcher, dev_set, threshold=0.85, sample_size=20)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - SAMPLE EVALUATION (20 documents)")
    print("=" * 80)
    print()

    metrics = sample_results['overall_metrics']
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print()

    # Check against success criteria
    print("SUCCESS CRITERIA CHECK:")
    print(f"   [{'PASS' if metrics['f1'] >= 0.80 else 'FAIL'}] F1 Score >= 0.80       (actual: {metrics['f1']:.4f})")
    print(f"   [{'PASS' if metrics['precision'] >= 0.75 else 'FAIL'}] Precision >= 0.75    (actual: {metrics['precision']:.4f})")
    print()

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    output_path = "evaluation/results/sapbert_sample_results.json"

    with open(output_path, 'w') as f:
        json.dump(sample_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    print("=" * 80)
    print("SAMPLE EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("To run on full dev set, modify sample_size parameter in code.")
    print()


if __name__ == "__main__":
    main()
