# -------------------------------------------------------
# Complete Statistical Analysis on Full Dataset
# -------------------------------------------------------

import json
import os
from collections import defaultdict, Counter


def load_json_file(path):
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_basic_stats(numbers):
    """Calculate mean, median, min, max."""
    if not numbers:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'count': 0}

    sorted_nums = sorted(numbers)
    n = len(sorted_nums)

    return {
        'mean': sum(numbers) / len(numbers),
        'median': sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2,
        'min': min(numbers),
        'max': max(numbers),
        'count': n
    }


def analyze_full_dataset():
    """Comprehensive dataset analysis on ALL data."""
    print("=" * 100)
    print("MeDiC COMPLETE DATASET STATISTICAL ANALYSIS")
    print("=" * 100)
    print()

    # Load synonym groups
    print("Loading synonym groups...")
    synonym_groups = load_json_file("data/synonym_groups.json")
    print(f"   Synonym groups loaded: {len(synonym_groups):,} concepts")
    print()

    # Load manual annotations
    print("Loading manual annotations...")
    manual_ann_path = "data/processed/manual_annotations.jsonl"
    manual_docs = []
    if os.path.exists(manual_ann_path):
        try:
            manual_docs = load_json_file(manual_ann_path)
            if not isinstance(manual_docs, list):
                manual_docs = [manual_docs]
        except:
            pass
        print(f"   Manual annotations loaded: {len(manual_docs):,} documents")
    print()

    # Load ALL MedMentions docs
    print("Loading ALL MedMentions documents...")
    mm_dir = "data/processed/MM_docs"
    mm_docs = []
    if os.path.exists(mm_dir):
        mm_files = [f for f in os.listdir(mm_dir) if f.endswith('.json')]
        print(f"   Total files found: {len(mm_files):,}")
        print(f"   Processing all documents (please wait)...")

        for i, filename in enumerate(mm_files, 1):
            if i % 500 == 0:
                print(f"      Progress: {i}/{len(mm_files)} ({i/len(mm_files)*100:.1f}%)")
            try:
                doc = load_json_file(os.path.join(mm_dir, filename))
                mm_docs.append(doc)
            except Exception as e:
                print(f"      Error loading {filename}: {e}")

        print(f"   Successfully loaded: {len(mm_docs):,} documents")
    print()

    # Statistics collection
    stats = {
        'total_docs_manual': len(manual_docs),
        'total_docs_mm': len(mm_docs),
        'total_annotations': 0,
        'unique_cuis': set(),
        'cui_frequency': Counter(),
        'synonym_type_dist': Counter(),
        'annotations_per_doc': [],
        'text_span_lengths': [],
        'synonyms_per_group': []
    }

    # Analyze manual annotations
    print("Analyzing manual annotations...")
    for doc in manual_docs:
        anns = doc.get('annotations', [])
        stats['annotations_per_doc'].append(len(anns))
        stats['total_annotations'] += len(anns)

        for ann in anns:
            cui = ann.get('canonical_concept')
            if cui:
                stats['unique_cuis'].add(cui)
                stats['cui_frequency'][cui] += 1

            syn_type = ann.get('synonym_type', 'unknown')
            stats['synonym_type_dist'][syn_type] += 1

            text = ann.get('annotated_text', '')
            stats['text_span_lengths'].append(len(text))

    print(f"   Processed {len(manual_docs):,} manual documents")
    print()

    # Analyze ALL MedMentions
    print("Analyzing ALL MedMentions documents...")
    for i, doc in enumerate(mm_docs, 1):
        if i % 500 == 0:
            print(f"   Progress: {i}/{len(mm_docs)} ({i/len(mm_docs)*100:.1f}%)")

        anns = doc.get('annotations', [])
        stats['annotations_per_doc'].append(len(anns))
        stats['total_annotations'] += len(anns)

        for ann in anns:
            cui = ann.get('canonical_concept')
            if cui:
                stats['unique_cuis'].add(cui)
                stats['cui_frequency'][cui] += 1

            syn_type = ann.get('synonym_type', 'unknown')
            stats['synonym_type_dist'][syn_type] += 1

            text = ann.get('annotated_text', '')
            stats['text_span_lengths'].append(len(text))

    print(f"   Processed {len(mm_docs):,} MedMentions documents")
    print()

    # Analyze synonym groups
    print("Analyzing synonym groups...")
    for cui, group_data in synonym_groups.items():
        synonyms = group_data.get('synonyms', [])
        stats['synonyms_per_group'].append(len(synonyms))
    print(f"   Processed {len(synonym_groups):,} synonym groups")
    print()

    # Print results
    print()
    print("=" * 100)
    print("RESULTS - COMPLETE DATASET ANALYSIS")
    print("=" * 100)
    print()

    print("DOCUMENT COUNTS:")
    print(f"   Manual annotations:        {stats['total_docs_manual']:,}")
    print(f"   MedMentions documents:     {stats['total_docs_mm']:,}")
    print(f"   Total documents analyzed:  {len(stats['annotations_per_doc']):,}")
    print()

    print("ANNOTATION STATISTICS:")
    print(f"   Total annotations:         {stats['total_annotations']:,}")
    print(f"   Unique CUIs:               {len(stats['unique_cuis']):,}")

    ann_stats = calculate_basic_stats(stats['annotations_per_doc'])
    print(f"   Annotations per document:")
    print(f"      Mean:                   {ann_stats['mean']:.2f}")
    print(f"      Median:                 {ann_stats['median']:.2f}")
    print(f"      Min:                    {ann_stats['min']}")
    print(f"      Max:                    {ann_stats['max']}")
    print()

    print("SYNONYM TYPE DISTRIBUTION:")
    total_types = sum(stats['synonym_type_dist'].values())
    for syn_type, count in stats['synonym_type_dist'].most_common():
        percentage = (count / total_types * 100) if total_types > 0 else 0
        print(f"   {syn_type:20s}  {count:7,} ({percentage:6.2f}%)")
    print()

    print("TEXT SPAN LENGTH STATISTICS:")
    span_stats = calculate_basic_stats(stats['text_span_lengths'])
    print(f"   Mean:                      {span_stats['mean']:.2f} characters")
    print(f"   Median:                    {span_stats['median']:.2f} characters")
    print(f"   Min:                       {span_stats['min']}")
    print(f"   Max:                       {span_stats['max']}")
    print()

    print("SYNONYM GROUP STATISTICS:")
    print(f"   Total groups:              {len(synonym_groups):,}")
    syn_stats = calculate_basic_stats(stats['synonyms_per_group'])
    print(f"   Synonyms per group:")
    print(f"      Mean:                   {syn_stats['mean']:.2f}")
    print(f"      Median:                 {syn_stats['median']:.2f}")
    print(f"      Min:                    {syn_stats['min']}")
    print(f"      Max:                    {syn_stats['max']}")
    print()

    print("TOP 30 MOST FREQUENT CONCEPTS:")
    for i, (cui, count) in enumerate(stats['cui_frequency'].most_common(30), 1):
        label = synonym_groups.get(cui, {}).get('canonical_label', 'Unknown')
        print(f"   {i:3d}. {cui}: {label[:60]:60s} ({count:6,})")
    print()

    # Calculate some basic metrics
    print("=" * 100)
    print("EVALUATION METRICS FRAMEWORK")
    print("=" * 100)
    print()

    # Example metrics calculation
    print("EXAMPLE: Precision, Recall, F1 Calculation")
    print()
    tp, fp, fn = 85, 10, 15  # Example values

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   True Positives:            {tp}")
    print(f"   False Positives:           {fp}")
    print(f"   False Negatives:           {fn}")
    print()
    print(f"   Precision = TP / (TP + FP) = {tp} / {tp + fp} = {precision:.4f}")
    print(f"   Recall    = TP / (TP + FN) = {tp} / {tp + fn} = {recall:.4f}")
    print(f"   F1 Score  = 2 * (P * R) / (P + R) = {f1:.4f}")
    print()

    print("SUCCESS CRITERIA (from proposal):")
    check_f1 = "PASS" if f1 >= 0.80 else "FAIL"
    check_prec = "PASS" if precision >= 0.75 else "FAIL"
    print(f"   [{check_f1}] F1 Score >= 0.80              (example: {f1:.4f})")
    print(f"   [{check_prec}] Precision >= 0.75            (example: {precision:.4f})")
    print(f"   [----] Recall improvement >= 20% over baseline")
    print(f"   [----] p95 latency < 200ms")
    print(f"   [----] Memory < 150MB")
    print()

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)

    output = {
        'document_counts': {
            'manual': stats['total_docs_manual'],
            'medmentions': stats['total_docs_mm'],
            'total': len(stats['annotations_per_doc'])
        },
        'annotation_statistics': {
            'total_annotations': stats['total_annotations'],
            'unique_cuis': len(stats['unique_cuis']),
            'annotations_per_doc': ann_stats
        },
        'synonym_type_distribution': dict(stats['synonym_type_dist']),
        'text_span_statistics': span_stats,
        'synonym_group_statistics': {
            'total_groups': len(synonym_groups),
            'synonyms_per_group': syn_stats
        },
        'top_30_concepts': [
            {'cui': cui, 'label': synonym_groups.get(cui, {}).get('canonical_label', 'Unknown'), 'count': count}
            for cui, count in stats['cui_frequency'].most_common(30)
        ],
        'example_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

    output_path = "evaluation/results/full_dataset_statistics.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    analyze_full_dataset()
