# -------------------------------------------------------
# Error Analysis Framework (dev set)
# 1. False positives - why incorrect matches were made
# 2. False negatives - why correct matches were missed
# 3. Error categorization:
#    - Lexical mismatches (surface form differences)
#    - Contextual errors (wrong sense disambiguation)
#    - Embedding drift (semantic similarity failures)
#    - Boundary errors (incorrect span boundaries)
# 4. Error patterns and insights for system improvement
# -------------------------------------------------------

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import difflib


class ErrorAnalyzer:
    """Analyze errors in detail from development set predictions."""

    def __init__(self, dev_set_path: str, synonym_groups_path: str):
        self.dev_set = self._load_dev_set(dev_set_path)
        self.synonym_groups = self._load_synonym_groups(synonym_groups_path)
        self.error_categories = {
            'lexical_mismatch': [],
            'contextual_error': [],
            'embedding_drift': [],
            'boundary_error': [],
            'abbreviation_expansion': [],
            'multi_word_error': [],
            'unknown_concept': []
        }

    def _load_dev_set(self, path: str) -> List[Dict]:
        """Load development set (JSONL or JSON)."""
        docs = []
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    docs = data
                else:
                    docs = [data]
        return docs

    def _load_synonym_groups(self, path: str) -> Dict:
        """Load synonym groups."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_false_positives(self, predicted_spans: List[Dict],
                                gold_spans: List[Dict],
                                document_text: str) -> List[Dict]:
        """
        Analyze false positive errors.

        Args:
            predicted_spans: Predicted annotations
            gold_spans: Ground truth annotations
            document_text: Original document text

        Returns:
            List of error analysis dicts
        """
        fp_errors = []

        # Build lookup for gold spans by position
        gold_lookup = {}
        for gold in gold_spans:
            key = (gold['start_offset'], gold['end_offset'])
            gold_lookup[key] = gold

        for pred in predicted_spans:
            pred_key = (pred['start_offset'], pred['end_offset'])

            # Check if this prediction matches any gold span
            if pred_key not in gold_lookup:
                # Check for overlapping gold spans (boundary errors)
                overlapping = self._find_overlapping_spans(pred, gold_spans)

                error_info = {
                    'type': 'false_positive',
                    'predicted_text': document_text[pred['start_offset']:pred['end_offset']],
                    'predicted_cui': pred.get('canonical_concept'),
                    'start': pred['start_offset'],
                    'end': pred['end_offset'],
                    'context': self._get_context(document_text, pred['start_offset'], pred['end_offset']),
                    'error_category': self._categorize_fp_error(pred, overlapping, document_text)
                }

                if overlapping:
                    error_info['overlapping_gold'] = [
                        {
                            'text': document_text[g['start_offset']:g['end_offset']],
                            'cui': g.get('canonical_concept'),
                            'overlap_ratio': self._calculate_overlap(pred, g)
                        }
                        for g in overlapping
                    ]

                fp_errors.append(error_info)
                self.error_categories[error_info['error_category']].append(error_info)

        return fp_errors

    def analyze_false_negatives(self, predicted_spans: List[Dict],
                                gold_spans: List[Dict],
                                document_text: str) -> List[Dict]:
        """
        Analyze false negative errors (missed annotations).

        Args:
            predicted_spans: Predicted annotations
            gold_spans: Ground truth annotations
            document_text: Original document text

        Returns:
            List of error analysis dicts
        """
        fn_errors = []

        # Build lookup for predicted spans
        pred_lookup = {}
        for pred in predicted_spans:
            key = (pred['start_offset'], pred['end_offset'])
            pred_lookup[key] = pred

        for gold in gold_spans:
            gold_key = (gold['start_offset'], gold['end_offset'])

            if gold_key not in pred_lookup:
                # Check for overlapping predictions (boundary errors)
                overlapping = self._find_overlapping_spans(gold, predicted_spans)

                cui = gold.get('canonical_concept')
                synonym_info = self.synonym_groups.get(cui, {})

                error_info = {
                    'type': 'false_negative',
                    'gold_text': document_text[gold['start_offset']:gold['end_offset']],
                    'gold_cui': cui,
                    'canonical_label': synonym_info.get('canonical_label', ''),
                    'synonyms': synonym_info.get('synonyms', []),
                    'start': gold['start_offset'],
                    'end': gold['end_offset'],
                    'context': self._get_context(document_text, gold['start_offset'], gold['end_offset']),
                    'error_category': self._categorize_fn_error(gold, overlapping, document_text, synonym_info)
                }

                if overlapping:
                    error_info['overlapping_predictions'] = [
                        {
                            'text': document_text[p['start_offset']:p['end_offset']],
                            'cui': p.get('canonical_concept'),
                            'overlap_ratio': self._calculate_overlap(gold, p)
                        }
                        for p in overlapping
                    ]

                fn_errors.append(error_info)
                self.error_categories[error_info['error_category']].append(error_info)

        return fn_errors

    def _find_overlapping_spans(self, target: Dict, spans: List[Dict],
                                threshold: float = 0.1) -> List[Dict]:
        """Find spans that overlap with target span."""
        overlapping = []
        for span in spans:
            overlap_ratio = self._calculate_overlap(target, span)
            if overlap_ratio > threshold:
                overlapping.append(span)
        return overlapping

    def _calculate_overlap(self, span1: Dict, span2: Dict) -> float:
        """Calculate overlap ratio between two spans."""
        start1, end1 = span1['start_offset'], span1['end_offset']
        start2, end2 = span2['start_offset'], span2['end_offset']

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_len = overlap_end - overlap_start
        max_len = max(end1 - start1, end2 - start2)

        return overlap_len / max_len if max_len > 0 else 0.0

    def _get_context(self, text: str, start: int, end: int,
                    context_window: int = 50) -> str:
        """Get surrounding context for an annotation."""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)

        context = text[context_start:context_end]
        # Mark the target span
        target_start = start - context_start
        target_end = end - context_start

        return (
            context[:target_start] +
            ">>>" + context[target_start:target_end] + "<<<" +
            context[target_end:]
        )

    def _categorize_fp_error(self, pred: Dict, overlapping: List[Dict],
                            text: str) -> str:
        """Categorize false positive error."""
        pred_text = text[pred['start_offset']:pred['end_offset']]

        # Boundary error: overlaps with gold but boundaries don't match
        if overlapping:
            return 'boundary_error'

        # Abbreviation expansion error
        if len(pred_text) <= 6 and pred_text.isupper():
            return 'abbreviation_expansion'

        # Multi-word error
        if ' ' in pred_text and len(pred_text.split()) > 3:
            return 'multi_word_error'

        # Unknown concept
        if pred.get('canonical_concept') == 'UNKNOWN' or not pred.get('canonical_concept'):
            return 'unknown_concept'

        # Default to contextual error
        return 'contextual_error'

    def _categorize_fn_error(self, gold: Dict, overlapping: List[Dict],
                            text: str, synonym_info: Dict) -> str:
        """Categorize false negative error."""
        gold_text = text[gold['start_offset']:gold['end_offset']]

        # Boundary error
        if overlapping:
            return 'boundary_error'

        # Lexical mismatch: surface form not in synonym list
        synonyms = synonym_info.get('synonyms', [])
        canonical = synonym_info.get('canonical_label', '')

        gold_text_lower = gold_text.lower()
        if gold_text_lower not in [s.lower() for s in synonyms] and gold_text_lower != canonical.lower():
            return 'lexical_mismatch'

        # Abbreviation
        if len(gold_text) <= 6 and gold_text.isupper():
            return 'abbreviation_expansion'

        # Multi-word
        if ' ' in gold_text and len(gold_text.split()) > 2:
            return 'multi_word_error'

        # Embedding drift (in synonym list but not found)
        if synonyms and gold_text_lower in [s.lower() for s in synonyms]:
            return 'embedding_drift'

        return 'unknown_concept'

    def generate_error_report(self, fp_errors: List[Dict],
                             fn_errors: List[Dict],
                             output_path: str):
        """Generate comprehensive error analysis report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("ERROR ANALYSIS REPORT - DEVELOPMENT SET")
        report_lines.append("=" * 100)
        report_lines.append("")

        # Summary statistics
        report_lines.append("📊 ERROR SUMMARY")
        report_lines.append(f"  Total False Positives: {len(fp_errors)}")
        report_lines.append(f"  Total False Negatives: {len(fn_errors)}")
        report_lines.append("")

        # Error category distribution
        report_lines.append("📈 ERROR CATEGORY DISTRIBUTION")
        for category, errors in self.error_categories.items():
            if errors:
                fp_count = sum(1 for e in errors if e['type'] == 'false_positive')
                fn_count = sum(1 for e in errors if e['type'] == 'false_negative')
                report_lines.append(f"  {category:25s}: {len(errors):4d} (FP: {fp_count}, FN: {fn_count})")
        report_lines.append("")

        # Detailed examples for each category
        report_lines.append("=" * 100)
        report_lines.append("DETAILED ERROR EXAMPLES (max 5 per category)")
        report_lines.append("=" * 100)
        report_lines.append("")

        for category, errors in self.error_categories.items():
            if errors:
                report_lines.append(f"\n{'='*100}")
                report_lines.append(f"CATEGORY: {category.upper().replace('_', ' ')}")
                report_lines.append(f"{'='*100}\n")

                for i, error in enumerate(errors[:5], 1):  # Show max 5 examples
                    report_lines.append(f"Example {i}:")
                    report_lines.append(f"  Type: {error['type']}")

                    if error['type'] == 'false_positive':
                        report_lines.append(f"  Predicted Text: '{error['predicted_text']}'")
                        report_lines.append(f"  Predicted CUI: {error['predicted_cui']}")
                    else:  # false_negative
                        report_lines.append(f"  Missed Text: '{error['gold_text']}'")
                        report_lines.append(f"  Gold CUI: {error['gold_cui']}")
                        report_lines.append(f"  Canonical Label: {error['canonical_label']}")

                    report_lines.append(f"  Context: {error['context']}")
                    report_lines.append("")

        # Save report
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n📄 Error analysis report saved to: {output_path}")

        # Also save as JSON for programmatic access
        json_output = {
            'summary': {
                'total_fp': len(fp_errors),
                'total_fn': len(fn_errors),
                'by_category': {cat: len(errs) for cat, errs in self.error_categories.items()}
            },
            'false_positives': fp_errors[:100],  # Save top 100
            'false_negatives': fn_errors[:100],
            'error_categories': {cat: errs[:20] for cat, errs in self.error_categories.items()}
        }

        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        print(f"📄 Error analysis JSON saved to: {json_path}")

        return report_text


def main():
    """Example usage of error analyzer."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS - DEVELOPMENT SET")
    print("=" * 80 + "\n")

    # Note: This requires actual predictions to compare against gold standard
    # For now, we'll set up the framework

    analyzer = ErrorAnalyzer(
        dev_set_path="data/doc_splits/dev.json",
        synonym_groups_path="data/synonym_groups.json"
    )

    print("✅ Error analyzer initialized")
    print(f"   Development set loaded: {len(analyzer.dev_set)} documents")
    print(f"   Synonym groups loaded: {len(analyzer.synonym_groups)} concepts")
    print("\n⚠️  Note: To run full error analysis, predictions must be generated first")
    print("   using a semantic matching model (SapBERT, MiniLM-Bio, etc.)")


if __name__ == "__main__":
    main()
