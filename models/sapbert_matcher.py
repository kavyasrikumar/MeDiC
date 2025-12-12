import json
import os
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Run: pip install torch transformers tqdm")
    raise

# Optional: SciSpacy for medical NER-based candidate extraction
try:
    import spacy
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False

# ------------------------------------------------------------------
# SapBERT semantic matcher for medical concept normalization.
# semantic matching implementation: Encodes medical terms, 
#   matches them against 34K concepts, finds semantic similarities
# ------------------------------------------------------------------

class SapBERTMatcher:
    """
    SapBERT semantic matcher for medical concept normalization.

    Key features:
    1. Predicts without seeing gold labels (no data leakage)
    2. Better candidate span extraction
    3. Length-based filtering to reduce false positives
    4. Efficient caching
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", use_ner: bool = True):
        """Initialize SapBERT matcher."""
        print("Initializing SapBERT matcher...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")

        print(f"   Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.concept_embeddings = {}  # CUI -> embedding
        self.concept_info = {}  # CUI -> {canonical_label, terms}
        self.cui_to_terms = defaultdict(set)  # CUI -> set of term strings

        # Load SciSpacy NER if available and requested
        self.nlp = None
        if use_ner and SCISPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                print("   Loaded SciSpacy NER model for candidate extraction")
            except:
                print("   SciSpacy model not found - using n-gram fallback")
                self.nlp = None

        print("   Model loaded successfully")

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([])

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # L2 normalize
                norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
                batch_embs = batch_embs / (norms + 1e-10)

                embeddings.append(batch_embs)

        return np.vstack(embeddings)

    def build_concept_index(self, synonym_groups_path: str):
        """Build concept embedding index from synonym groups."""
        print("\nBuilding concept embedding index...")

        with open(synonym_groups_path, 'r') as f:
            synonym_groups = json.load(f)

        print(f"   Loaded {len(synonym_groups):,} concept groups")

        # Collect all unique terms and their CUIs
        all_terms = []
        term_to_cui = {}

        for cui, group_data in synonym_groups.items():
            canonical_label = group_data.get('canonical_label', '')
            synonyms = group_data.get('synonyms', [])

            self.concept_info[cui] = {
                'canonical_label': canonical_label,
                'terms': [canonical_label] + synonyms
            }

            # Add all terms
            for term in [canonical_label] + synonyms:
                term_lower = term.lower().strip()
                if term_lower and len(term_lower) > 1:  # Filter very short terms
                    all_terms.append(term_lower)
                    term_to_cui[term_lower] = cui
                    self.cui_to_terms[cui].add(term_lower)

        # Get unique terms
        unique_terms = list(set(all_terms))
        print(f"   Collected {len(unique_terms):,} unique terms")
        print(f"   Encoding to embeddings (this may take a while)...")

        # Encode all terms
        embeddings = self.encode_texts(unique_terms, batch_size=64)

        # Aggregate embeddings per CUI (average of all synonyms)
        cui_embeddings = defaultdict(list)
        for term, emb in zip(unique_terms, embeddings):
            cui = term_to_cui[term]
            cui_embeddings[cui].append(emb)

        # Average embeddings for each CUI
        for cui, embs in cui_embeddings.items():
            avg_emb = np.mean(embs, axis=0)
            # Re-normalize
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            self.concept_embeddings[cui] = avg_emb

        print(f"   Index built: {len(self.concept_embeddings):,} concept embeddings")

    def _extract_candidate_spans(self, text: str, max_span_words: int = 5) -> List[Dict]:
        """
        Extract candidate text spans with MINIMAL filtering.

        Strategy: Extract liberally, rely on SapBERT semantic matching + confidence filtering.
        """
        import re

        # Only ultra-common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        words = text.split()
        candidates = []

        # Extract ALL n-grams (1-5 words)
        for n in range(1, max_span_words + 1):
            for i in range(len(words) - n + 1):
                span_words = words[i:i+n]
                span_text = ' '.join(span_words).strip()

                # MINIMAL filtering - only reject obviously bad spans
                if len(span_text) < 3:  # Too short (single/double letters)
                    continue
                if n == 1 and span_words[0].lower() in stop_words:  # Single stop word
                    continue
                if span_text.isdigit():  # Pure number
                    continue
                if re.match(r'^[\d\s\.,;:!\?-]+$', span_text):  # Only punctuation/digits
                    continue

                # Find position
                start_pos = text.find(span_text)
                if start_pos != -1:
                    candidates.append({
                        'text': span_text,
                        'start': start_pos,
                        'end': start_pos + len(span_text),
                        'num_words': n
                    })

        # Remove substring duplicates (keep longest spans)
        filtered = []
        candidates.sort(key=lambda x: len(x['text']), reverse=True)
        for cand in candidates:
            is_substring = False
            for kept in filtered:
                if (cand['start'] >= kept['start'] and cand['end'] <= kept['end']):
                    is_substring = True
                    break
            if not is_substring:
                filtered.append(cand)

        return filtered

    def _extract_candidates_with_ner(self, text: str) -> List[Dict]:
        """
        Extract candidate spans using SciSpacy NER.

        Much more accurate than n-grams - only extracts actual medical entities.
        """
        if not self.nlp:
            # Fallback to n-gram method
            return self._extract_candidate_spans(text)

        doc = self.nlp(text)
        candidates = []

        for ent in doc.ents:
            candidates.append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'num_words': len(ent.text.split()),
                'entity_type': ent.label_
            })

        return candidates

    def predict_on_document(self, document: Dict, threshold: float = 0.85,
                           max_predictions: int = 1000,
                           min_confidence_gap: float = 0.05) -> List[Dict]:
        """
        Generate predictions for a document WITHOUT using gold annotations.

        Args:
            document: Document dict with 'text' field
            threshold: Similarity threshold (higher = more precise)
            max_predictions: Maximum predictions to return
            min_confidence_gap: Minimum gap between 1st and 2nd best match (reduces false positives)

        Returns:
            List of predicted annotations
        """
        text = document.get('text', '')

        if not text or not self.concept_embeddings:
            return []

        # Extract candidate spans using NER if available, otherwise fall back to n-grams
        if self.nlp:
            candidates = self._extract_candidates_with_ner(text)
        else:
            candidates = self._extract_candidate_spans(text, max_span_words=5)

        if not candidates:
            return []

        # Encode candidates
        candidate_texts = [c['text'].lower() for c in candidates]
        candidate_embeddings = self.encode_texts(candidate_texts, batch_size=64)

        # Convert concept embeddings to matrix
        cuis = list(self.concept_embeddings.keys())
        concept_matrix = np.vstack([self.concept_embeddings[cui] for cui in cuis])

        # Calculate similarities: candidates x concepts
        # Shape: (num_candidates, num_concepts)
        similarities = np.dot(candidate_embeddings, concept_matrix.T)

        # For each candidate, find best matching concept
        predictions = []
        for i, candidate in enumerate(candidates):
            # Get top 2 best matching concepts
            top_indices = np.argsort(similarities[i])[-2:][::-1]  # Top 2 in descending order
            best_sim = similarities[i, top_indices[0]]
            second_best_sim = similarities[i, top_indices[1]] if len(top_indices) > 1 else 0.0

            # Require: (1) above threshold AND (2) confident (gap from 2nd best)
            confidence_gap = best_sim - second_best_sim
            if best_sim >= threshold and confidence_gap >= min_confidence_gap:
                best_cui = cuis[top_indices[0]]

                predictions.append({
                    'text_span_id': f"pred_{len(predictions):04d}",
                    'start_offset': candidate['start'],
                    'end_offset': candidate['end'],
                    'annotated_text': candidate['text'],
                    'canonical_concept': best_cui,
                    'similarity_score': float(best_sim),
                    'confidence_gap': float(confidence_gap),
                    'num_words': candidate['num_words']
                })

        # Remove overlapping spans (keep higher confidence one)
        predictions = self._remove_overlaps(predictions)

        # Sort by similarity and limit
        predictions.sort(key=lambda x: x['similarity_score'], reverse=True)
        predictions = predictions[:max_predictions]

        return predictions

    def _remove_overlaps(self, predictions: List[Dict]) -> List[Dict]:
        """
        Remove overlapping predictions, keeping the one with higher similarity.
        If spans are nested, keep the longer one if similarities are similar.
        """
        if not predictions:
            return []

        # Sort by similarity (descending)
        sorted_preds = sorted(predictions, key=lambda x: x['similarity_score'], reverse=True)

        kept = []
        for pred in sorted_preds:
            # Check if this prediction overlaps with any kept prediction
            overlaps = False
            for kept_pred in kept:
                # Check for overlap
                if not (pred['end_offset'] <= kept_pred['start_offset'] or
                        pred['start_offset'] >= kept_pred['end_offset']):
                    overlaps = True
                    break

            if not overlaps:
                kept.append(pred)

        return kept

    def save_embeddings(self, output_path: str):
        """Save precomputed embeddings."""
        output = {
            'concept_embeddings': {cui: emb.tolist() for cui, emb in self.concept_embeddings.items()},
            'concept_info': self.concept_info
        }

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f)

        print(f"   Embeddings saved to: {output_path}")

    def load_embeddings(self, input_path: str):
        """Load precomputed embeddings."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        self.concept_embeddings = {cui: np.array(emb) for cui, emb in data['concept_embeddings'].items()}
        self.concept_info = data.get('concept_info', {})

        # Rebuild cui_to_terms
        for cui, info in self.concept_info.items():
            for term in info.get('terms', []):
                self.cui_to_terms[cui].add(term.lower())

        print(f"   Loaded {len(self.concept_embeddings):,} concept embeddings")
