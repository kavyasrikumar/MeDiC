"""
Analyze what entities the NER is missing
=========================================
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

import spacy

def main():
    print("=" * 80)
    print("ANALYZING MISSED ENTITIES")
    print("=" * 80)
    print()

    # Load NER model
    nlp = spacy.load("en_core_sci_sm")

    # Load dev set
    with open("data/kaggle_splits/dev.json", 'r') as f:
        docs = json.load(f)[:50]
    
    print(f"Analyzing {len(docs)} documents...")
    print()

    # For each document, check which gold entities are detected by NER
    total_gold = 0
    detected = 0
    missed_examples = []
    missed_lengths = Counter()

    for doc in docs:
        text = doc.get('text', '')
        gold_anns = doc.get('annotations', [])
        
        # Run NER
        spacy_doc = nlp(text)
        ner_spans = [(ent.start_char, ent.end_char, ent.text) for ent in spacy_doc.ents]
        
        for ann in gold_anns:
            total_gold += 1
            gold_start = ann.get('start_offset', 0)
            gold_end = ann.get('end_offset', 0)
            gold_text = ann.get('annotated_text', '')
            
            # Check if NER detected something overlapping
            is_detected = False
            for ner_start, ner_end, ner_text in ner_spans:
                # Check overlap
                overlap_start = max(gold_start, ner_start)
                overlap_end = min(gold_end, ner_end)
                if overlap_end > overlap_start:
                    is_detected = True
                    break
            
            if is_detected:
                detected += 1
            else:
                # Track missed entity
                word_count = len(gold_text.split())
                missed_lengths[word_count] += 1
                if len(missed_examples) < 50:
                    missed_examples.append({
                        'text': gold_text,
                        'cui': ann.get('canonical_concept', ''),
                        'word_count': word_count
                    })

    print(f"Total gold entities: {total_gold}")
    print(f"Detected by NER: {detected} ({100*detected/total_gold:.1f}%)")
    print(f"Missed by NER: {total_gold - detected} ({100*(total_gold-detected)/total_gold:.1f}%)")
    print()

    print("Missed entities by word count:")
    for wc in sorted(missed_lengths.keys()):
        print(f"   {wc} words: {missed_lengths[wc]}")
    print()

    print("Sample missed entities (first 30):")
    for ex in missed_examples[:30]:
        print(f"   '{ex['text']}' (CUI: {ex['cui']}, {ex['word_count']} words)")

    # Analyze what types of terms are missed
    print()
    print("=" * 80)
    print("PATTERN ANALYSIS OF MISSED ENTITIES")
    print("=" * 80)
    
    # Common patterns in missed entities
    single_word_missed = [ex for ex in missed_examples if ex['word_count'] == 1]
    multi_word_missed = [ex for ex in missed_examples if ex['word_count'] > 1]
    
    print(f"\nSingle-word missed entities ({len(single_word_missed)}):")
    for ex in single_word_missed[:15]:
        print(f"   '{ex['text']}'")
    
    print(f"\nMulti-word missed entities ({len(multi_word_missed)}):")
    for ex in multi_word_missed[:15]:
        print(f"   '{ex['text']}'")


if __name__ == "__main__":
    main()

