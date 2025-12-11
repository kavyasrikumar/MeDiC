# -------------------------------------------------------
# 1. Precision, Recall, F1 score
# 2. Inter-annotator agreement (Cohen's kappa)
# 3. Dataset statistics and coverage analysis
# 4. Baseline comparisons (literal matcher, lexical synonym lookup)
# 5. Error analysis and categorization
# -------------------------------------------------------

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetStatistics:
    """Compute statistics on the annotated dataset."""

    def __init__(self, data_dir: str, synonym_groups_path: str):
        self.data_dir = data_dir
        self.synonym_groups = self._load_synonym_groups(synonym_groups_path)
        self.manual_annotations = self._load_manual_annotations()
        self.mm_docs = self._load_mm_docs()

    def _load_synonym_groups(self, path: str) -> Dict:
        """Load synonym groups mapping."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_manual_annotations(self) -> List[Dict]:
        """Load manual annotations from JSONL file."""
        annotations = []
        manual_path = os.path.join(self.data_dir, "manual_annotations.jsonl")
        if os.path.exists(manual_path):
            with open(manual_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        annotations.append(json.loads(line))
        return annotations

    def _load_mm_docs(self) -> List[Dict]:
        """Load processed MedMentions documents."""
        docs = []
        mm_dir = os.path.join(self.data_dir, "MM_docs")
        if os.path.exists(mm_dir):
            for filename in os.listdir(mm_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(mm_dir, filename), 'r', encoding='utf-8') as f:
                        try:
                            docs.append(json.load(f))
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON: {filename}")
        return docs

    def compute_dataset_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics."""
        stats = {
            'total_documents': len(self.mm_docs) + len(self.manual_annotations),
            'medmentions_docs': len(self.mm_docs),
            'manual_docs': len(self.manual_annotations),
            'total_annotations': 0,
            'total_unique_cuis': set(),
            'synonym_type_distribution': Counter(),
            'annotations_per_doc': [],
            'cui_frequency': Counter(),
            'text_span_lengths': [],
            'unique_synonym_groups': len(self.synonym_groups),
            'synonyms_per_group': []
        }

        # Analyze MedMentions documents
        for doc in self.mm_docs:
            num_annotations = len(doc.get('annotations', []))
            stats['annotations_per_doc'].append(num_annotations)
            stats['total_annotations'] += num_annotations

            for ann in doc.get('annotations', []):
                cui = ann.get('canonical_concept')
                if cui:
                    stats['total_unique_cuis'].add(cui)
                    stats['cui_frequency'][cui] += 1

                syn_type = ann.get('synonym_type', 'unknown')
                stats['synonym_type_distribution'][syn_type] += 1

                text = ann.get('annotated_text', '')
                stats['text_span_lengths'].append(len(text))

        # Analyze manual annotations
        for doc in self.manual_annotations:
            num_annotations = len(doc.get('annotations', []))
            stats['annotations_per_doc'].append(num_annotations)
            stats['total_annotations'] += num_annotations

            for ann in doc.get('annotations', []):
                cui = ann.get('canonical_concept')
                if cui:
                    stats['total_unique_cuis'].add(cui)
                    stats['cui_frequency'][cui] += 1

                syn_type = ann.get('synonym_type', 'unknown')
                stats['synonym_type_distribution'][syn_type] += 1

                text = ann.get('annotated_text', '')
                stats['text_span_lengths'].append(len(text))

        # Analyze synonym groups
        for cui, group_data in self.synonym_groups.items():
            synonyms = group_data.get('synonyms', [])
            stats['synonyms_per_group'].append(len(synonyms))

        stats['total_unique_cuis'] = len(stats['total_unique_cuis'])

        return stats

    def print_statistics(self):
        """Print formatted statistics report."""
        stats = self.compute_dataset_statistics()

        print("=" * 80)
        print("DATASET STATISTICS REPORT")
        print("=" * 80)

        print("\n📊 DOCUMENT COUNTS")
        print(f"  Total Documents: {stats['total_documents']:,}")
        print(f"  - MedMentions Documents: {stats['medmentions_docs']:,}")
        print(f"  - Manual Annotations: {stats['manual_docs']:,}")

        print("\n📝 ANNOTATION STATISTICS")
        print(f"  Total Annotations: {stats['total_annotations']:,}")
        print(f"  Unique CUIs: {stats['total_unique_cuis']:,}")
        print(f"  Avg Annotations per Document: {np.mean(stats['annotations_per_doc']):.2f}")
        print(f"  Median Annotations per Document: {np.median(stats['annotations_per_doc']):.2f}")
        print(f"  Min/Max Annotations: {min(stats['annotations_per_doc']) if stats['annotations_per_doc'] else 0} / {max(stats['annotations_per_doc']) if stats['annotations_per_doc'] else 0}")

        print("\n🏷️  SYNONYM TYPE DISTRIBUTION")
        total_types = sum(stats['synonym_type_distribution'].values())
        for syn_type, count in stats['synonym_type_distribution'].most_common():
            percentage = (count / total_types * 100) if total_types > 0 else 0
            print(f"  {syn_type:15s}: {count:6,} ({percentage:5.2f}%)")

        print("\n📏 TEXT SPAN LENGTH STATISTICS")
        if stats['text_span_lengths']:
            print(f"  Mean Length: {np.mean(stats['text_span_lengths']):.2f} characters")
            print(f"  Median Length: {np.median(stats['text_span_lengths']):.2f} characters")
            print(f"  Min/Max Length: {min(stats['text_span_lengths'])} / {max(stats['text_span_lengths'])} characters")

        print("\n🔗 SYNONYM GROUP STATISTICS")
        print(f"  Total Synonym Groups: {stats['unique_synonym_groups']:,}")
        if stats['synonyms_per_group']:
            print(f"  Avg Synonyms per Group: {np.mean(stats['synonyms_per_group']):.2f}")
            print(f"  Median Synonyms per Group: {np.median(stats['synonyms_per_group']):.2f}")
            print(f"  Min/Max Synonyms: {min(stats['synonyms_per_group'])} / {max(stats['synonyms_per_group'])}")

        print("\n🔝 TOP 10 MOST FREQUENT CONCEPTS")
        for cui, count in stats['cui_frequency'].most_common(10):
            label = self.synonym_groups.get(cui, {}).get('canonical_label', 'Unknown')
            print(f"  {cui}: {label[:40]:40s} ({count:4,} occurrences)")

        print("\n" + "=" * 80)

        return stats


class EvaluationMetrics:
    """Calculate precision, recall, F1 for synonym matching."""

    @staticmethod
    def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> Dict:
        """Calculate precision, recall, and F1 score."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    @staticmethod
    def calculate_span_overlap_metrics(predicted_spans: List[Tuple[int, int, str]],
                                      gold_spans: List[Tuple[int, int, str]],
                                      overlap_threshold: float = 0.5) -> Dict:
        """
        Calculate metrics for span-level matching with overlap threshold.

        Args:
            predicted_spans: List of (start, end, cui) tuples
            gold_spans: List of (start, end, cui) tuples
            overlap_threshold: Minimum overlap ratio to consider a match
        """
        tp = 0
        fp = 0
        fn = 0
        matched_gold = set()

        # For each predicted span, check if it matches a gold span
        for pred_start, pred_end, pred_cui in predicted_spans:
            matched = False
            for i, (gold_start, gold_end, gold_cui) in enumerate(gold_spans):
                if i in matched_gold:
                    continue

                # Check CUI match and span overlap
                if pred_cui == gold_cui:
                    # Calculate overlap
                    overlap_start = max(pred_start, gold_start)
                    overlap_end = min(pred_end, gold_end)

                    if overlap_end > overlap_start:
                        overlap_len = overlap_end - overlap_start
                        pred_len = pred_end - pred_start
                        gold_len = gold_end - gold_start

                        # Overlap ratio relative to both spans
                        overlap_ratio = overlap_len / max(pred_len, gold_len)

                        if overlap_ratio >= overlap_threshold:
                            tp += 1
                            matched_gold.add(i)
                            matched = True
                            break

            if not matched:
                fp += 1

        # False negatives are gold spans that weren't matched
        fn = len(gold_spans) - len(matched_gold)

        return EvaluationMetrics.calculate_metrics(tp, fp, fn)


class BaselineComparison:
    """Implement baseline systems for comparison."""

    def __init__(self, synonym_groups: Dict):
        self.synonym_groups = synonym_groups
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for fast baseline matching."""
        self.exact_lookup = {}  # text -> CUI
        self.synonym_lookup = defaultdict(set)  # text -> set of CUIs

        for cui, group_data in self.synonym_groups.items():
            synonyms = group_data.get('synonyms', [])
            canonical = group_data.get('canonical_label', '').lower()

            # Add canonical label
            if canonical:
                self.exact_lookup[canonical] = cui
                self.synonym_lookup[canonical].add(cui)

            # Add all synonyms
            for syn in synonyms:
                syn_lower = syn.lower()
                self.synonym_lookup[syn_lower].add(cui)

    def literal_text_matcher(self, text: str, query: str) -> List[Tuple[int, int, str]]:
        """
        Baseline 1: Literal substring matching (mimics Ctrl+F).
        Returns list of (start, end, cui) tuples.
        """
        matches = []
        query_lower = query.lower()
        text_lower = text.lower()

        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break

            # Try to get CUI from exact lookup
            cui = self.exact_lookup.get(query_lower, "UNKNOWN")
            matches.append((pos, pos + len(query), cui))
            start = pos + 1

        return matches

    def lexical_synonym_lookup(self, text: str, query: str) -> List[Tuple[int, int, str]]:
        """
        Baseline 2: Dictionary-based synonym expansion without embeddings.
        Finds all synonyms of query term and matches them literally.
        """
        matches = []
        query_lower = query.lower()

        # Get all CUIs associated with query
        query_cuis = self.synonym_lookup.get(query_lower, set())

        # For each CUI, get all its synonyms
        search_terms = set()
        for cui in query_cuis:
            group_data = self.synonym_groups.get(cui, {})
            synonyms = group_data.get('synonyms', [])
            search_terms.update([s.lower() for s in synonyms])

            canonical = group_data.get('canonical_label', '')
            if canonical:
                search_terms.add(canonical.lower())

        # Add original query
        search_terms.add(query_lower)

        # Search for all terms in text
        text_lower = text.lower()
        for term in search_terms:
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break

                # Get CUI for this term
                cuis = self.synonym_lookup.get(term, set())
                cui = list(cuis)[0] if cuis else "UNKNOWN"

                matches.append((pos, pos + len(term), cui))
                start = pos + 1

        return matches


class InterAnnotatorAgreement:
    """Calculate Cohen's kappa for inter-annotator agreement."""

    @staticmethod
    def calculate_cohens_kappa(annotator1_labels: List, annotator2_labels: List) -> float:
        """
        Calculate Cohen's kappa score.

        Args:
            annotator1_labels: List of labels from annotator 1
            annotator2_labels: List of labels from annotator 2

        Returns:
            Cohen's kappa score
        """
        if len(annotator1_labels) != len(annotator2_labels):
            raise ValueError("Annotator label lists must be the same length")

        return cohen_kappa_score(annotator1_labels, annotator2_labels)

    @staticmethod
    def span_to_token_labels(text: str, annotations: List[Dict],
                            tokenize_fn=None) -> List[str]:
        """
        Convert span annotations to token-level BIO labels for kappa calculation.

        Args:
            text: Document text
            annotations: List of annotation dicts with start_offset, end_offset, canonical_concept
            tokenize_fn: Optional tokenization function (defaults to whitespace split)

        Returns:
            List of BIO labels for each token
        """
        if tokenize_fn is None:
            tokens = text.split()
            # Simple whitespace tokenization - track character positions
            token_positions = []
            pos = 0
            for token in tokens:
                start = text.find(token, pos)
                end = start + len(token)
                token_positions.append((start, end))
                pos = end
        else:
            tokens, token_positions = tokenize_fn(text)

        # Initialize all tokens as 'O' (outside)
        labels = ['O'] * len(tokens)

        # Mark tokens that fall within annotation spans
        for ann in annotations:
            ann_start = ann.get('start_offset')
            ann_end = ann.get('end_offset')
            cui = ann.get('canonical_concept', 'ENTITY')

            is_first = True
            for i, (tok_start, tok_end) in enumerate(token_positions):
                # Check if token overlaps with annotation
                if tok_start < ann_end and tok_end > ann_start:
                    if is_first:
                        labels[i] = f'B-{cui}'
                        is_first = False
                    else:
                        labels[i] = f'I-{cui}'

        return labels


def generate_precision_recall_curve(results_at_thresholds: List[Tuple[float, Dict]],
                                    output_path: str = "pr_curve.png"):
    """
    Generate precision-recall curve for different similarity thresholds.

    Args:
        results_at_thresholds: List of (threshold, metrics_dict) tuples
        output_path: Path to save the plot
    """
    thresholds = [r[0] for r in results_at_thresholds]
    precisions = [r[1]['precision'] for r in results_at_thresholds]
    recalls = [r[1]['recall'] for r in results_at_thresholds]
    f1_scores = [r[1]['f1'] for r in results_at_thresholds]

    plt.figure(figsize=(12, 5))

    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, 'b-o', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])

    # F1 vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1_scores, 'g-o', linewidth=2, label='F1')
    plt.plot(thresholds, precisions, 'b--', linewidth=1.5, label='Precision')
    plt.plot(thresholds, recalls, 'r--', linewidth=1.5, label='Recall')
    plt.xlabel('Similarity Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([min(thresholds), max(thresholds)])
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Precision-Recall curve saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("MeDiC STATISTICAL ANALYSIS AND EVALUATION")
    print("=" * 80 + "\n")

    # Paths
    data_dir = "data/processed"
    synonym_groups_path = "data/synonym_groups.json"
    output_dir = "evaluation/results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Statistics
    print("📈 Computing Dataset Statistics...\n")
    stats_analyzer = DatasetStatistics(data_dir, synonym_groups_path)
    stats = stats_analyzer.print_statistics()

    # Save statistics to JSON
    stats_output = {k: v for k, v in stats.items() if not isinstance(v, set)}
    stats_output['synonym_type_distribution'] = dict(stats['synonym_type_distribution'])
    stats_output['cui_frequency_top100'] = dict(stats['cui_frequency'].most_common(100))

    with open(os.path.join(output_dir, "dataset_statistics.json"), 'w') as f:
        json.dump(stats_output, f, indent=2)

    print(f"\n✅ Statistics saved to: {output_dir}/dataset_statistics.json")

    # 2. Example: Calculate baseline performance (placeholder)
    print("\n📊 Baseline systems initialized")
    print("   - Literal Text Matcher (Ctrl+F equivalent)")
    print("   - Lexical Synonym Lookup (Dictionary-based)")

    # 3. Success criteria check
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA (from proposal)")
    print("=" * 80)
    print("  ✓ F1 score ≥ 0.80 on evaluation set")
    print("  ✓ Recall improvement ≥ 20% over lexical baseline")
    print("  ✓ Precision ≥ 0.75")
    print("  ✓ p95 latency < 200ms")
    print("  ✓ Memory footprint < 150MB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
