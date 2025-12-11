"""
Test BC5CDR NER model for better entity detection
==================================================
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import spacy

# Check models
print("Available NER models:")
for name in ['en_core_sci_sm', 'en_ner_bc5cdr_md']:
    try:
        nlp = spacy.load(name)
        labels = list(nlp.pipe_labels.get('ner', []))
        print(f"  ✓ {name}: {labels}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
print()


class MultiModelMatcher:
    """Matcher that combines multiple NER models."""
    
    def __init__(self, base_matcher, model_names=['en_core_sci_sm', 'en_ner_bc5cdr_md']):
        self.matcher = base_matcher
        self.models = []
        for name in model_names:
            try:
                nlp = spacy.load(name)
                self.models.append((name, nlp))
                print(f"  Loaded {name}")
            except:
                print(f"  Could not load {name}")
    
    def extract_entities(self, text):
        """Extract entities from all models, merged."""
        all_entities = []
        
        for name, nlp in self.models:
            doc = nlp(text)
            for ent in doc.ents:
                all_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'source': name
                })
        
        # Merge overlapping entities (keep longest)
        merged = []
        all_entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        for ent in all_entities:
            overlaps = False
            for kept in merged:
                if not (ent['end'] <= kept['start'] or ent['start'] >= kept['end']):
                    overlaps = True
                    break
            if not overlaps:
                merged.append(ent)
        
        return merged
    
    def predict(self, doc, threshold=0.70, gap=0.02):
        """Predict using merged NER entities."""
        text = doc.get('text', '')
        if not text:
            return []
        
        # Extract entities from all models
        entities = self.extract_entities(text)
        
        if not entities:
            return []
        
        # Encode and match with SapBERT
        entity_texts = [e['text'].lower() for e in entities]
        embeddings = self.matcher.encode_texts(entity_texts, batch_size=64)
        
        cuis = list(self.matcher.concept_embeddings.keys())
        concept_matrix = np.vstack([self.matcher.concept_embeddings[cui] for cui in cuis])
        similarities = np.dot(embeddings, concept_matrix.T)
        
        predictions = []
        for i, ent in enumerate(entities):
            top_indices = np.argsort(similarities[i])[-2:][::-1]
            best_sim = similarities[i, top_indices[0]]
            second_sim = similarities[i, top_indices[1]] if len(top_indices) > 1 else 0.0
            conf_gap = best_sim - second_sim
            
            if best_sim >= threshold and conf_gap >= gap:
                best_cui = cuis[top_indices[0]]
                predictions.append({
                    'start_offset': ent['start'],
                    'end_offset': ent['end'],
                    'annotated_text': ent['text'],
                    'canonical_concept': best_cui,
                    'similarity_score': float(best_sim),
                    'entity_type': ent['label'],
                    'source': ent['source']
                })
        
        return self._remove_overlaps(predictions)
    
    def _remove_overlaps(self, predictions):
        if not predictions:
            return []
        sorted_preds = sorted(predictions, key=lambda x: x['similarity_score'], reverse=True)
        kept = []
        for pred in sorted_preds:
            overlaps = False
            for kept_pred in kept:
                if not (pred['end_offset'] <= kept_pred['start_offset'] or 
                        pred['start_offset'] >= kept_pred['end_offset']):
                    overlaps = True
                    break
            if not overlaps:
                kept.append(pred)
        return kept


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def compare_predictions(predicted_anns, gold_anns, overlap_threshold=0.5):
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


def evaluate(matcher, docs, threshold, gap):
    total_tp, total_fp, total_fn = 0, 0, 0
    for doc in docs:
        preds = matcher.predict(doc, threshold=threshold, gap=gap)
        gold = doc.get('annotations', [])
        tp, fp, fn = compare_predictions(preds, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return calculate_metrics(total_tp, total_fp, total_fn)


def main():
    from models.sapbert_matcher import SapBERTMatcher
    
    print("=" * 80)
    print("TESTING MULTI-MODEL NER APPROACH")
    print("=" * 80)
    print()

    # Load base matcher
    print("Loading SapBERT matcher...")
    base_matcher = SapBERTMatcher(use_ner=False)  # Don't load default NER
    base_matcher.load_embeddings("models/sapbert_embeddings.json")
    print()

    # Load dev set
    print("Loading development set...")
    with open("data/kaggle_splits/dev.json", 'r') as f:
        docs = json.load(f)[:50]
    print(f"   Loaded {len(docs)} documents")
    print()

    # Test different model combinations
    print("=" * 80)
    print("COMPARING NER MODEL COMBINATIONS")
    print("=" * 80)
    print()
    print(f"{'Model':>40} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    # Single model: en_core_sci_sm
    print("\nSingle model tests:")
    matcher1 = MultiModelMatcher(base_matcher, ['en_core_sci_sm'])
    m = evaluate(matcher1, docs, threshold=0.70, gap=0.02)
    print(f"{'en_core_sci_sm only':>40} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Single model: en_ner_bc5cdr_md  
    matcher2 = MultiModelMatcher(base_matcher, ['en_ner_bc5cdr_md'])
    m = evaluate(matcher2, docs, threshold=0.70, gap=0.02)
    print(f"{'en_ner_bc5cdr_md only':>40} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Both models combined
    print("\nCombined models:")
    matcher3 = MultiModelMatcher(base_matcher, ['en_core_sci_sm', 'en_ner_bc5cdr_md'])
    m = evaluate(matcher3, docs, threshold=0.70, gap=0.02)
    print(f"{'Both models combined':>40} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Try different thresholds with combined
    print("\nTuning combined model:")
    for t, g in [(0.70, 0.02), (0.65, 0.02), (0.70, 0.01), (0.65, 0.01)]:
        m = evaluate(matcher3, docs, threshold=t, gap=g)
        print(f"{f'Combined T={t} G={g}':>40} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")


if __name__ == "__main__":
    main()

