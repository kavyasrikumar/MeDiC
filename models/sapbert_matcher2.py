# sapbert_matcher_fixed.py
import json
import os
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    import torch
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except Exception as e:
    raise ImportError("Install transformers, torch and tqdm: pip install transformers torch tqdm") from e

# Try to import FAISS (optional but highly recommended)
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
    # we'll fall back to batched dot-products


def normalize_text(s: str) -> str:
    """Lowercase, unicode normalize, remove extra punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    # replace punctuation with space but keep alphanum and percent/plus/etc if needed
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class SapBERTMatcher:
    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 device: Optional[str] = None,
                 use_faiss: bool = True):
        print("Initializing SapBERTMatcher")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f" Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.concept_embeddings: Dict[str, np.ndarray] = {}   # CUI -> embedding
        self.concept_info: Dict[str, dict] = {}               # CUI -> meta (canonical_label, terms)
        self.term_to_cuis: Dict[str, List[str]] = defaultdict(list)  # term -> [CUIs]
        self.cui_to_terms: Dict[str, List[str]] = defaultdict(list)

        self.faiss_index = None
        self.cui_list = []   # index -> CUI
        self.use_faiss = use_faiss and HAVE_FAISS

        print(" Model and tokenizer loaded.")

    # -----------------------
    # Encoding utilities
    # -----------------------
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of strings to L2-normalized embeddings (numpy array)."""
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                print(f"  Encoding progress: {i}/{len(texts)} ({(i/len(texts))*100:.1f}%)")
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # CLS embedding
                embs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                # L2 normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)
                all_embs.append(embs)
        import multiprocessing
        try:
            multiprocessing.resource_tracker._CLEANUP_SEMAPHORES = True
        except AttributeError:
            pass
        return np.vstack(all_embs)

    # -----------------------
    # Build index
    # -----------------------
    def build_concept_index(self, synonym_groups_path: str, encode_batch_size: int = 64, filter_short: int = 2,
                        checkpoint_path: Optional[str] = None, checkpoint_interval: int = 500):
        """
        Builds embeddings aggregated per CUI using only canonical concepts (no synonyms) and an ANN index.
        Optionally saves intermediate embeddings to checkpoint_path every checkpoint_interval concepts.

        Expected synonym_groups_path format: dict keyed by CUI:
        {
        "C001": {"canonical_label": "Heatstroke", "synonyms": ["heat stroke", "heatstroke"]},
        ...
        }
        """
        print("Loading synonym groups:", synonym_groups_path)
        with open(synonym_groups_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Extract canonical terms only, filter short
        canonical_terms = []
        cui_list = []
        for cui, info in data.items():
            canonical = info.get("canonical_label") or info.get("canonical_concept") or ""
            normalized = normalize_text(canonical)
            if len(normalized) > filter_short:
                canonical_terms.append(normalized)
                cui_list.append(cui)

        print(f"Collected {len(canonical_terms)} canonical terms for embedding")

        # If checkpoint exists, try loading it to resume
        embeddings = []
        start_idx = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint embeddings from {checkpoint_path} ...")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                embeddings = [np.array(vec, dtype=np.float32) for vec in checkpoint_data["embeddings"]]
                start_idx = len(embeddings)
                print(f"Resuming from index {start_idx}")

        # Encode remaining canonical terms in batches
        with torch.no_grad():
            for i in range(start_idx, len(canonical_terms), encode_batch_size):
                batch = canonical_terms[i:i+encode_batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)
                embeddings.extend(embs)

                print(f"Encoded {min(i+encode_batch_size, len(canonical_terms))}/{len(canonical_terms)} canonical terms")

                # Save checkpoint periodically
                if checkpoint_path and (i // encode_batch_size) % checkpoint_interval == 0:
                    print(f"Saving checkpoint to {checkpoint_path} ...")
                    checkpoint_out = {
                        "embeddings": [emb.tolist() for emb in embeddings]
                    }
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint_out, f)

        # Final save after encoding all
        if checkpoint_path:
            print(f"Final save checkpoint to {checkpoint_path} ...")
            checkpoint_out = {
                "embeddings": [emb.tolist() for emb in embeddings]
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_out, f)

        # Aggregate embeddings per CUI (here each CUI has exactly one canonical term embedding)
        self.concept_embeddings = {cui: emb for cui, emb in zip(cui_list, embeddings)}
        self.cui_list = cui_list

        # Store concept info for canonical terms only
        for cui, term in zip(cui_list, canonical_terms):
            self.concept_info[cui] = {"canonical_label": term, "terms": [term]}

        print(f"Built {len(self.concept_embeddings)} canonical CUI embeddings.")

        # Build FAISS index or fallback
        matrix = np.vstack(embeddings).astype(np.float32)

        if self.use_faiss:
            dim = matrix.shape[1]
            print("Building FAISS index (IndexFlatIP) for inner-product search...")
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(matrix)
            print("FAISS index built with", self.faiss_index.ntotal, "vectors.")
        else:
            self._concept_matrix = matrix
            print("FAISS not available — will use batched dot-product fallback.")

    # -----------------------
    # Candidate span extraction (robust)
    # -----------------------
    def _extract_candidate_spans(self, text: str, max_span_words: int = 5, min_chars: int = 3):
        """Fast and safe: extracts all contiguous spans up to max_span_words."""
        if not text:
            return []

        tokens = text.split()
        spans = []
        char_offsets = []

        # Precompute char offsets for tokens
        cur = 0
        for tok in tokens:
            start = text.find(tok, cur)
            end = start + len(tok)
            char_offsets.append((start, end))
            cur = end

        # Sliding window of up to max_span_words
        for i in range(len(tokens)):
            for n in range(1, max_span_words + 1):
                if i + n > len(tokens):
                    break

                span_tokens = tokens[i:i+n]
                span_text = " ".join(span_tokens)

                if len(span_text) < min_chars:
                    continue

                start = char_offsets[i][0]
                end = char_offsets[i+n-1][1]

                spans.append({
                    "text": span_text,
                    "start_offset": start,
                    "end_offset": end,
                    "num_words": n
                })

        return spans

    # -----------------------
    # Prediction
    # -----------------------
    def predict_on_document(self, document: Dict, threshold: float = 0.7, top_k: int = 5, max_span_words: int = 5) -> List[Dict]:
        """
        For each candidate span, return ranked candidate CUIs with scores.
        Returns list of predictions where each prediction has:
          start_offset, end_offset, annotated_text, candidates: [{'cui','score'}, ...]
        # """
        # Handle document as a list
        if isinstance(document, list):
            text = " ".join(str(item) for item in document)
        elif isinstance(document, dict):
            text = document.get("text", "") or document.get("full_text", "")
        else:
            text = str(document)
        
        text = text.strip() if text else ""
        if not text or not self.concept_embeddings:
            return []

        print("Extracting candidate spans...")
        candidates = self._extract_candidate_spans(text, max_span_words=max_span_words)
        if not candidates:
            return []

        # encode candidate texts (normalized)
        cand_texts = [normalize_text(c["text"]) for c in candidates]
        print("Encoding candidate spans...")
        cand_embs = self.encode_texts(cand_texts, batch_size=32)  # normalized and l2

        results = []
        # Use FAISS if available for top-k search; otherwise batched dot product
        if self.use_faiss and self.faiss_index is not None:
            # search (inner product on normalized vectors approximates cosine)
            D, I = self.faiss_index.search(cand_embs.astype(np.float32), top_k)
            for idx, cand in enumerate(candidates):
                print(f"  Processing candidate {idx + 1}/{len(candidates)} ({(idx + 1) / len(candidates) * 100:.1f}%)")
                cand_scores = D[idx]
                cand_idxs = I[idx]
                ranked = []
                for score, i in zip(cand_scores, cand_idxs):
                    cui = self.cui_list[i]
                    ranked.append({"cui": cui, "score": float(score)})
                # apply threshold on top score
                if ranked and ranked[0]["score"] >= threshold:
                    results.append({
                        "start_offset": cand["start_offset"],
                        "end_offset": cand["end_offset"],
                        "annotated_text": cand["text"],
                        "candidates": ranked
                    })
        else:
            # fallback: compute dot-products in batches
            matrix = getattr(self, "_concept_matrix", None)
            if matrix is None:
                matrix = np.vstack([self.concept_embeddings[cui] for cui in self.cui_list]).astype(np.float32)
                self._concept_matrix = matrix
            # batch over candidates
            B = 256
            for i0 in range(0, len(cand_embs), B):
                batch = cand_embs[i0:i0+B]
                sims = np.dot(batch, self._concept_matrix.T)  # (b, n_concepts)
                for j in range(batch.shape[0]):
                    row = sims[j]
                    top_idx = np.argsort(row)[-top_k:][::-1]
                    ranked = [{"cui": self.cui_list[k], "score": float(row[k])} for k in top_idx]
                    if ranked and ranked[0]["score"] >= threshold:
                        cand = candidates[i0 + j]
                        results.append({
                            "start_offset": cand["start_offset"],
                            "end_offset": cand["end_offset"],
                            "annotated_text": cand["text"],
                            "candidates": ranked
                        })

        # Sort results by candidate top score (descending)
        results.sort(key=lambda r: (r["candidates"][0]["score"] if r["candidates"] else 0.0), reverse=True)
        print(f"Generated {len(results)} predictions above threshold {threshold}")
        return results

    # -----------------------
    # Save / Load embeddings
    # -----------------------
    def save_embeddings(self, output_path: str):
        data = {
            "concept_embeddings": {cui: emb.tolist() for cui, emb in self.concept_embeddings.items()},
            "concept_info": self.concept_info,
            "cui_list": self.cui_list
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        print("Saved embeddings to", output_path)

    def load_embeddings(self, input_path: str):
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.concept_embeddings = {cui: np.array(vec, dtype=np.float32) for cui, vec in data["concept_embeddings"].items()}
        self.concept_info = data.get("concept_info", {})
        self.cui_list = list(self.concept_embeddings.keys())
        # build fallback matrix
        self._concept_matrix = np.vstack([self.concept_embeddings[c] for c in self.cui_list]).astype(np.float32)
        if self.use_faiss:
            dim = self._concept_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self._concept_matrix)
        print("Loaded embeddings; total CUIs:", len(self.concept_embeddings))