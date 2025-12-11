"""
Improved SapBERT Matcher - No Data Leakage
===========================================

Fixes critical issue: predict_on_document was using gold annotations as queries.
This version properly predicts by matching all text spans against all concepts.
"""

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


class SapBERTMatcher:
    """
    SapBERT semantic matcher for medical concept normalization.

    Key features:
    1. Predicts without seeing gold labels (no data leakage)
    2. Better candidate span extraction
    3. Length-based filtering to reduce false positives
    4. Efficient caching
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
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
        Extract candidate text spans with better filtering.

        Improvements:
        - Skip very short spans (< 3 chars)
        - Skip pure numbers
        - Skip common stop words
        """
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

        words = text.split()
        candidates = []
        seen = set()  # Avoid duplicate spans

        for n in range(1, max_span_words + 1):
            for i in range(len(words) - n + 1):
                span_words = words[i:i+n]
                span_text = ' '.join(span_words).strip()

                # Filter out bad candidates
                if len(span_text) < 3:  # Too short
                    continue
                if span_text.lower() in stop_words:  # Stop word
                    continue
                if span_text.isdigit():  # Pure number
                    continue
                if span_text in seen:  # Duplicate
                    continue

                # Find character positions
                start_pos = text.find(span_text)
                if start_pos != -1:
                    candidates.append({
                        'text': span_text,
                        'start': start_pos,
                        'end': start_pos + len(span_text),
                        'num_words': n
                    })
                    seen.add(span_text)

        return candidates

    def predict_on_document(self, document: Dict, threshold: float = 0.85,
                           max_predictions: int = 1000) -> List[Dict]:
        """
        Generate predictions for a document WITHOUT using gold annotations.

        Args:
            document: Document dict with 'text' field
            threshold: Similarity threshold (higher = more precise)
            max_predictions: Maximum predictions to return

        Returns:
            List of predicted annotations
        """
        text = document.get('text', '')

        if not text or not self.concept_embeddings:
            return []

        # Extract candidate spans
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
            # Get best matching concept
            best_concept_idx = np.argmax(similarities[i])
            best_sim = similarities[i, best_concept_idx]

            if best_sim >= threshold:
                best_cui = cuis[best_concept_idx]

                predictions.append({
                    'text_span_id': f"pred_{len(predictions):04d}",
                    'start_offset': candidate['start'],
                    'end_offset': candidate['end'],
                    'annotated_text': candidate['text'],
                    'canonical_concept': best_cui,
                    'similarity_score': float(best_sim),
                    'num_words': candidate['num_words']
                })

        # Sort by similarity and limit
        predictions.sort(key=lambda x: x['similarity_score'], reverse=True)
        predictions = predictions[:max_predictions]

        return predictions

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
