#!/usr/bin/env python3
"""
medmentions_normalize.py

Normalize MedMentions -> your annotation schema + build synonym groups (hybrid using MRCONSO).
Optionally override/augment with existing manual annotations.

Usage:
  python medmentions_normalize.py
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict, OrderedDict
import textwrap
import hashlib

# -------------------------
# Helpers: MedMentions parsing
# -------------------------
def load_medmentions_pubtator(pubtator_path):
    """
    Parse PubTator-style MedMentions file into dict pmid -> {text, annotations[]}
    Annotation rows expected as tab-separated:
      pmid<TAB>start<TAB>end<TAB>span<TAB>semtypes<TAB>cui
    Title/abstract lines expected as:
      pmid|t|Title...
      pmid|a|Abstract...
    """
    docs = defaultdict(lambda: {"title": "", "abstract": "", "annotations": []})
    with open(pubtator_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Detect title/abstract lines (have '|' delimiters)
            if '|' in line and (line.count("|") >= 2):
                parts = line.split("|", 2)
                if len(parts) == 3 and parts[1] in ('t', 'a'):
                    pmid, part, text = parts
                    if part == 't':
                        docs[pmid]['title'] = text
                    elif part == 'a':
                        docs[pmid]['abstract'] = text
                    continue
            # Otherwise try tab-separated annotation row
            cols = line.split("\t")
            if len(cols) >= 6:
                pmid = cols[0]
                try:
                    start = int(cols[1])
                    end = int(cols[2])
                except ValueError:
                    continue
                span_text = cols[3]
                sem_types = cols[4].split(",") if cols[4] else []
                cui = cols[5]
                docs[pmid]['annotations'].append({
                    "start_offset": start,
                    "end_offset": end,
                    "text": span_text,
                    "semantic_types": sem_types,
                    "cui": cui
                })
            # else ignore
    # Build full text field
    for pmid, doc in docs.items():
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        doc['text'] = (title + "\n" + abstract).strip()
        doc.pop('title', None)
        doc.pop('abstract', None)
    return docs

# -------------------------
# MRCONSO handling (zipped)
# -------------------------
def ensure_mrconso_unzipped(zip_path, extract_dir):
    """
    If zip_path exists, extract MRCONSO.RRF into extract_dir if not already present.
    Returns path to MRCONSO.RRF or None if not available.
    """
    mrconso_target = os.path.join(extract_dir, "MRCONSO.RRF")
    if os.path.isfile(mrconso_target):
        return mrconso_target
    if not zip_path:
        return None
    if not os.path.isfile(zip_path):
        print(f"[WARN] MRCONSO zip not found at {zip_path}. Skipping MRCONSO load.")
        return None
    os.makedirs(extract_dir, exist_ok=True)
    print(f"[INFO] Extracting MRCONSO from {zip_path} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # find MRCONSO.RRF inside zip (could be nested)
        candidate = None
        for name in zf.namelist():
            if name.endswith("MRCONSO.RRF"):
                candidate = name
                break
        if candidate:
            zf.extract(candidate, path=extract_dir)
            extracted = os.path.join(extract_dir, candidate)
            # move to root extract_dir if nested path
            if os.path.isfile(extracted) and os.path.dirname(extracted) != os.path.abspath(extract_dir):
                os.replace(extracted, mrconso_target)
        else:
            # extract all and attempt to find MRCONSO.RRF
            zf.extractall(path=extract_dir)
    # search for MRCONSO.RRF if not at target
    if not os.path.isfile(mrconso_target):
        for root, _, files in os.walk(extract_dir):
            if "MRCONSO.RRF" in files:
                mrconso_target = os.path.join(root, "MRCONSO.RRF")
                break
    if os.path.isfile(mrconso_target):
        print(f"[INFO] MRCONSO available at {mrconso_target}")
        return mrconso_target
    print("[WARN] MRCONSO.RRF not found after extraction.")
    return None

def parse_mrconso(mrconso_path, source_vocab_filter=None, language="ENG"):
    """
    Parse MRCONSO.RRF -> returns:
      cui_to_synonyms: dict CUI -> set(synonym strings)
      cui_to_preferred: dict CUI -> preferred label (if found)
    Filtering by source_vocab_filter (list of SABs) and language code.
    """
    cui_to_synonyms = defaultdict(set)
    cui_to_preferred = {}
    if not mrconso_path:
        return cui_to_synonyms, cui_to_preferred
    print(f"[INFO] Parsing MRCONSO.RRF (this may take a while)...")
    with open(mrconso_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue
            cui = parts[0]
            lat = parts[1]
            ispref = parts[6]   # sometimes 'Y' or '1' or 'Y'
            sab = parts[11]
            tty = parts[12]
            term = parts[14].strip()
            if not term:
                continue
            if language and lat != language:
                continue
            if source_vocab_filter and sab not in source_vocab_filter:
                continue
            term_l = term.lower()
            cui_to_synonyms[cui].add(term_l)
            # prefer PT (preferred term) or ISPREF marker if present
            if cui not in cui_to_preferred:
                if tty == "PT" or ispref in ("Y", "y", "1"):
                    cui_to_preferred[cui] = term
            else:
                # keep existing unless we find a PT or ispref true and preferred not set to PT earlier
                if (tty == "PT" or ispref in ("Y", "y", "1")) and cui_to_preferred.get(cui, "") == "":
                    cui_to_preferred[cui] = term
    # if some CUIs have no preferred, pick arbitrary first synonym as canonical label
    for cui, syns in cui_to_synonyms.items():
        if cui not in cui_to_preferred:
            # pick longest or first
            cand = sorted(syns, key=lambda s: (-len(s), s))[0] if syns else None
            if cand:
                cui_to_preferred[cui] = cand
    print(f"[INFO] Parsed MRCONSO: {len(cui_to_synonyms)} unique surface terms mapped across CUIs.")
    return cui_to_synonyms, cui_to_preferred

# -------------------------
# Manual annotations loader (optional overrides)
# -------------------------
def load_manual_annotations(manual_dir):
    """
    Load existing manual annotation JSON files (one or many) into a dict:
      manual_map[document_id] = {"annotations": [ ... ] }
    Manual files expected to follow your schema (annotation list with text_span_id etc).
    """
    manual_map = {}
    if not manual_dir or not os.path.isdir(manual_dir):
        return manual_map
    for fname in os.listdir(manual_dir):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(manual_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                j = json.load(fh)
                doc_id = j.get("document_id") or os.path.splitext(fname)[0]
                manual_map[doc_id] = j
        except Exception as e:
            print(f"[WARN] Could not load manual annotation {path}: {e}")
    print(f"[INFO] Loaded manual annotations for {len(manual_map)} documents from {manual_dir}")
    return manual_map

# -------------------------
# Normalization: build synonym groups (hybrid) and annotate each MedMentions annotation to your schema
# -------------------------
def make_synonym_groups(cui_to_synonyms, cui_to_preferred, docs, manual_map=None):
    """
    Build global synonym_groups mapping keyed by CUI, with group_id, canonical_concept, synonyms list.
    Use MRCONSO synonyms when available; if not, fallback to occurrences in MedMentions and manual annotations.
    manual_map: overrides. If manual has a group for a CUI, use it.
    """
    synonym_groups = {}
    group_counter = 1

    # If manual_map provided, collect manual-defined groups first
    manual_cui_to_groupid = {}
    if manual_map:
        for docid, mdoc in manual_map.items():
            for ann in mdoc.get("annotations", []):
                cui = ann.get("canonical_concept")
                gid = ann.get("synonym_group_id")
                if cui and gid:
                    # build group entry if not present
                    if cui not in synonym_groups:
                        synonyms = set()
                        # include annotated_text if present
                        if ann.get("annotated_text"):
                            synonyms.add(ann["annotated_text"].lower())
                        synonym_groups[cui] = {
                            "synonym_group_id": gid,
                            "canonical_concept": cui,
                            "synonyms": sorted(list(synonyms))
                        }
                        manual_cui_to_groupid[cui] = gid
                        # ensure group_counter won't clash (parse numeric suffix if possible)
                        try:
                            n = int(gid.split("_")[-1])
                            group_counter = max(group_counter, n+1)
                        except:
                            pass

    # Now build groups for CUIs seen in docs
    seen_cuis = set()
    for pmid, doc in docs.items():
        for ann in doc.get("annotations", []):
            cui = ann.get("cui")
            if not cui or cui in synonym_groups:
                continue
            seen_cuis.add(cui)
            # prefer MRCONSO synonyms
            synset = sorted(cui_to_synonyms.get(cui, [])) if cui_to_synonyms else []
            # include manual synonyms if manual_map has doc-level synonyms for this cui
            if manual_map:
                # collect any annotated_texts from manual docs that match CUI
                for mdoc in manual_map.values():
                    for mann in mdoc.get("annotations", []):
                        if mann.get("canonical_concept") == cui and mann.get("annotated_text"):
                            synset.append(mann["annotated_text"].lower())
            # fallback to the annotation text occurrences if no MRCONSO synonyms found
            if not synset:
                # collect text occurrences in docs
                occ = set()
                for pmid2, doc2 in docs.items():
                    for ann2 in doc2.get("annotations", []):
                        if ann2.get("cui") == cui and ann2.get("text"):
                            occ.add(ann2.get("text").lower())
                synset = sorted(list(occ)) if occ else []
            if not synset:
                synset = []
            # canonical label: use preferred if available, else first synonym, else empty
            canonical_label = cui_to_preferred.get(cui) if cui_to_preferred else None
            if canonical_label:
                canonical_label = canonical_label if isinstance(canonical_label, str) else str(canonical_label)
            elif synset:
                canonical_label = synset[0]
            else:
                canonical_label = ""

            group_id = f"syn_{group_counter:04d}"
            synonym_groups[cui] = {
                "synonym_group_id": group_id,
                "canonical_concept": cui,
                "canonical_label": canonical_label,
                "synonyms": sorted(list(dict.fromkeys([s.lower() for s in synset if s])))  # unique preserving order
            }
            group_counter += 1

    return synonym_groups

def heuristic_synonym_type(annotated_text, synonyms_set):
    """
    Heuristic for synonym_type:
      - if annotated_text.lower() exactly matches one of synonyms_set -> 'exact'
      - elif looks like abbreviation (all upper or short alpha) -> 'abbreviation'
      - else -> 'variant'
    """
    if not annotated_text:
        return "variant"
    at = annotated_text.strip()
    if not at:
        return "variant"
    lat = at.lower()
    if synonyms_set and lat in synonyms_set:
        return "exact"
    # simple abbreviation heuristic
    stripped = at.replace(".", "").replace("-", "")
    if stripped.isupper() and 1 < len(stripped) <= 6:
        return "abbreviation"
    # if it's very short (1-4 chars) and contains letters -> abbreviation
    if 1 < len(stripped) <= 4 and any(c.isalpha() for c in stripped) and stripped.upper() == stripped:
        return "abbreviation"
    return "variant"

# -------------------------
# Convert docs -> normalized schema and save
# -------------------------
def normalize_docs_and_save(docs, synonym_groups, manual_map, outdir):
    """
    For each doc in docs, convert each annotation to your schema and save per-doc JSON.
    If manual_map contains an overriding doc, use annotated fields from manual doc (canonical_label, etc).
    """
    docs_outdir = os.path.join(outdir, "docs")
    os.makedirs(docs_outdir, exist_ok=True)
    count = 0
    for pmid, doc in docs.items():
        doc_text = doc.get("text", "")
        anns = doc.get("annotations", [])
        normalized = []
        # If manual doc exists for this pmid, use manual annotations mapping by offsets or text match
        manual_doc = manual_map.get(pmid)
        # Build quick lookup of manual by (start,end) -> annotation
        manual_lookup = {}
        if manual_doc:
            for mann in manual_doc.get("annotations", []):
                key = (mann.get("start_offset"), mann.get("end_offset"))
                manual_lookup[key] = mann

        # generate span ids
        for i, ann in enumerate(anns, start=1):
            start = ann.get("start_offset")
            end = ann.get("end_offset")
            span_text = ann.get("text")
            cui = ann.get("cui") or ""
            sem_types = ann.get("semantic_types") or []
            # lookup synonym group
            sg = synonym_groups.get(cui, {}) if synonym_groups else {}
            sgid = sg.get("synonym_group_id")
            synonyms_set = set(sg.get("synonyms", [])) if sg else set()
            canonical_label = sg.get("canonical_label") or ""
            # If manual annotation exists for exact offsets, prefer manual fields
            man = manual_lookup.get((start, end))
            text_span_id = None
            if man:
                # prefer manual text_span_id if present, else create
                text_span_id = man.get("text_span_id") or f"span{str(i).zfill(4)}"
                annotated_text = man.get("annotated_text") or span_text
                canonical_concept = man.get("canonical_concept") or cui
                canonical_label = man.get("canonical_label") or canonical_label
                synonym_group_id = man.get("synonym_group_id") or sgid
                synonym_type = man.get("synonym_type") or heuristic_synonym_type(annotated_text, synonyms_set)
                notes = man.get("notes") or ("semantic types: " + ",".join(sem_types) if sem_types else "")
            else:
                text_span_id = f"span{str(i).zfill(4)}"
                annotated_text = span_text
                canonical_concept = cui
                synonym_group_id = sgid
                synonym_type = heuristic_synonym_type(annotated_text, synonyms_set)
                notes = "semantic types: " + ",".join(sem_types) if sem_types else ""
            normalized.append({
                "text_span_id": text_span_id,
                "start_offset": start,
                "end_offset": end,
                "annotated_text": annotated_text,
                "canonical_concept": canonical_concept,
                "canonical_label": canonical_label,
                "synonym_group_id": synonym_group_id,
                "synonym_type": synonym_type,
                "notes": notes
            })
        # save doc
        out_obj = {
            "document_id": pmid,
            "text": doc_text,
            "annotations": normalized
        }
        out_path = os.path.join(docs_outdir, f"{pmid}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, indent=2, ensure_ascii=False)
        count += 1
    print(f"[SAVE] Wrote {count} normalized documents to {docs_outdir}")

def save_synonym_groups_file(synonym_groups, outpath):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    # Convert synonym_groups keyed by CUI to a list or keep as dict keyed by CUI
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(synonym_groups, fh, indent=2, ensure_ascii=False)
    print(f"[SAVE] Wrote synonym groups to {outpath}")

# -------------------------
# CLI and main flow
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MedMentions -> normalized annotation schema with MRCONSO hybrid synonym groups")
    p.add_argument("--vocab-filter", nargs="*", default=None, help="Filter MRCONSO by SAB vocabularies (e.g., SNOMEDCT_US ICD10CM RXNORM).")
    p.add_argument("--limit", type=int, default=None, help="Limit number of documents processed (for quick testing).")
    return p.parse_args()

def main():
    args = parse_args()

    # STEP 1: load MedMentions
    docs = load_medmentions_pubtator("corpus_pubtator.txt")
    print(f"[INFO] Loaded {len(docs)} documents from corpus_pubtator.txt")

    # optional limit
    if args.limit:
        docs = dict(list(docs.items())[:args.limit])
        print(f"[INFO] Limiting to {len(docs)} documents for quick run")

    # STEP 2: MRCONSO extraction & parse
    mrconso_path = ensure_mrconso_unzipped("MRCONSO.RRF.zip", "data") if "MRCONSO.RRF.zip" else None
    cui_to_synonyms, cui_to_preferred = ({}, {}) if not mrconso_path else parse_mrconso(mrconso_path, source_vocab_filter=args.vocab_filter)

    # STEP 3: load manual annotations (optional)
    manual_map = load_manual_annotations("data/raw") if "data/raw" else {}

    # STEP 4: build hybrid synonym groups
    synonym_groups = make_synonym_groups(cui_to_synonyms, cui_to_preferred, docs, manual_map)

    print(f"[INFO] Built {len(synonym_groups)} synonym groups (hybrid MRCONSO + occurrences + manual overrides).")

    # STEP 5: normalize docs and save
    os.makedirs("data/processed", exist_ok=True)
    normalize_docs_and_save(docs, synonym_groups, manual_map, "data/processed")
    save_synonym_groups_file(synonym_groups, os.path.join("data/processed", "synonym_groups.json"))

    print("[DONE] All outputs saved. You can now annotate further or train models using the normalized JSONs.")

if __name__ == "__main__":
    main()