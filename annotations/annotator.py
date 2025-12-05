#!/usr/bin/env python3
"""
annotator.py

Usage:
    python3 annotator.py --text infile.txt --out annotations.json --lexicons lex1.csv lex2.csv

If no lexicons provided, a small built-in cardiology lexicon will be used.

Lexicon CSV format (optional; you can download from Kaggle/UMLS and convert to this):
    surface_form,canonical_id,synonym_type
Examples:
    heart attack,C0027051,preferred
    myocardial infarction,C0027051,exact
    angina,C0002962,preferred
"""

import argparse
import json
import csv
import re
import uuid
from collections import defaultdict, Counter

# -------------------------
# Small fallback cardiology lexicon (surface -> (cui, type))
# You can replace/extend this by passing --lexicons file(s)
# -------------------------
FALLBACK_LEXICON = [
    ("heart attack", "C0027051", "preferred"),
    ("myocardial infarction", "C0027051", "exact"),
    ("chest pain", "", "exact"),
    ("angina", "", "preferred"),
    ("atherosclerosis", "", "preferred"),
    ("plaque", "", "preferred"),
    ("clot", "", "preferred"),
    ("aspirin", "", "preferred"),
    ("ecg", "", "abbrev"),
    ("ekg", "", "abbrev"),
    ("cardiopulmonary resuscitation", "", "preferred"),
    ("cpr", "", "abbrev"),
    ("coronary artery disease", "", "preferred"),
    ("coronary arteries", "", "preferred"),
    ("coronary artery", "", "preferred"),
    ("heart muscle", "", "preferred"),
    ("blood clot", "", "exact"),
    ("sudden cardiac arrest", "", "preferred"),
    ("angina", "", "preferred"),
    ("plaque buildup", "", "preferred"),
    ("cholesterol", "", "preferred"),
    ("fatigue", "", "preferred"),
    ("shortness of breath", "", "preferred"),
    ("aspirin", "", "preferred"),
    ("aed", "", "abbrev"),
    ("automated external defibrillator", "", "preferred"),
    ("st elevation", "", "preferred"),
    ("ecg", "", "abbrev"),
    ("ekg", "", "abbrev")
]

# -------------------------
# Utilities
# -------------------------
def load_lexicons(filepaths):
    lex = []
    for p in filepaths:
        with open(p, newline='', encoding='utf8') as csvf:
            reader = csv.reader(csvf)
            for row in reader:
                if not row: continue
                # accept either 1,2 or 3 column rows
                surface = row[0].strip()
                cui = row[1].strip() if len(row) > 1 else ""
                stype = row[2].strip() if len(row) > 2 else "exact"
                if surface:
                    lex.append((surface.lower(), cui, stype))
    return lex

def build_trie(lexicon_entries):
    """
    Build a simple dict-of-dicts trie for longest-match scanning.
    Entries: list of (surface, cui, type)
    """
    trie = {}
    for surface, cui, stype in lexicon_entries:
        toks = surface.split()
        node = trie
        for t in toks:
            if t not in node:
                node[t] = {}
            node = node[t]
        node["_end"] = {"surface": surface, "cui": cui, "type": stype}
    return trie

def find_longest_matches(text, trie):
    """
    Find all non-overlapping occurrences by scanning tokens left-to-right and choosing longest matches.
    Returns list of dicts: {start_char, end_char, text, cui, type}
    """
    # tokenization that preserves positions: split on whitespace but we need char offsets
    pattern = re.compile(r"\S+")
    matches = []
    # Build a list of (token, start, end) for scanning
    toks = [(m.group(0), m.start(), m.end()) for m in pattern.finditer(text)]
    n = len(toks)
    i = 0
    while i < n:
        # attempt to walk trie for longest match starting at i
        node = trie
        j = i
        last_hit = None
        last_j = None
        while j < n:
            tok = toks[j][0].lower()
            # strip punctuation at edges for matching key tokens (but keep original spans)
            key = tok.strip("()[]{}.,;:\"'`")
            if key in node:
                node = node[key]
                if "_end" in node:
                    last_hit = node["_end"]
                    last_j = j
                j += 1
            else:
                break
        if last_hit:
            # compute char offsets from first token start to last token end
            start_char = toks[i][1]
            end_char = toks[last_j][2]
            surfaced = text[start_char:end_char]
            matches.append({
                "start": start_char,
                "end": end_char,
                "text": surfaced,
                "surface_key": last_hit["surface"],
                "cui": last_hit.get("cui",""),
                "syn_type": last_hit.get("type","exact")
            })
            i = last_j + 1
        else:
            i += 1
    return matches

# -------------------------
# Main annotator routine
# -------------------------
def annotate_text(text, lexicon_entries):
    # build trie
    trie = build_trie(lexicon_entries)
    raw_matches = find_longest_matches(text, trie)

    # normalize and group duplicate spans (if exact same offsets)
    span_map = {}
    for m in raw_matches:
        key = (m["start"], m["end"])
        if key not in span_map:
            span_map[key] = {
                "start": m["start"],
                "end": m["end"],
                "text": m["text"],
                "surfaces": set([m["surface_key"]]),
                "cuis": set([m["cui"]]) if m["cui"] else set(),
                "types": set([m["syn_type"]])
            }
        else:
            span_map[key]["surfaces"].add(m["surface_key"])
            if m["cui"]:
                span_map[key]["cuis"].add(m["cui"])
            span_map[key]["types"].add(m["syn_type"])

    # Assign unique span ids and synonym groups
    annotations = []
    synonym_groups = {}
    surface_to_group = {}
    group_counter = 1
    for (start,end), info in sorted(span_map.items(), key=lambda x: x[0]):
        sid = f"span{len(annotations)+1}"
        surfaces = sorted(info["surfaces"])
        cui = next(iter(info["cuis"])) if info["cuis"] else ""
        # create or reuse synonym group for this cui/surfaces
        # use canonical group id if same cui exists
        group_id = None
        if cui:
            # reuse group for same cui if created
            for gid, g in synonym_groups.items():
                if g["canonical_concept"] == cui:
                    group_id = gid
                    break
        if group_id is None:
            group_id = f"syn_{group_counter:04d}"
            group_counter += 1
            synonym_groups[group_id] = {
                "canonical_concept": cui,
                "synonyms": []
            }
        # add surfaces to synonym group
        for s in surfaces:
            if s not in [x["text"] for x in synonym_groups[group_id]["synonyms"]]:
                synonym_groups[group_id]["synonyms"].append({"text": s, "type": "exact"})

        annotations.append({
            "text_span_id": sid,
            "start_offset": info["start"],
            "end_offset": info["end"],
            "annotated_text": info["text"],
            "canonical_concept": cui,
            "synonym_group_id": group_id,
            "synonym_type": ",".join(sorted(info["types"])),
            "notes": ""
        })

    # produce frequency counts
    freq = Counter([a["annotated_text"].lower() for a in annotations])

    return annotations, synonym_groups, freq

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Plain text input file")
    parser.add_argument("--out", default="annotations.json", help="Output JSON file")
    parser.add_argument("--synout", default="synonym_groups.json", help="Synonym groups output")
    parser.add_argument("--lexicons", nargs="*", help="CSV lexicon files (surface,cui,type)")
    args = parser.parse_args()

    with open(args.text, 'r', encoding='utf8') as f:
        text = f.read()

    lex_entries = []
    if args.lexicons:
        lex_entries = load_lexicons(args.lexicons)
    else:
        lex_entries = [(s.lower(), cui, stype) for s,cui,stype in FALLBACK_LEXICON]

    annotations, syn_groups, freq = annotate_text(text, lex_entries)

    doc = {
        "document_id": "doc_" + str(uuid.uuid4())[:8],
        "text": text,
        "annotations": annotations
    }

    with open(args.out, 'w', encoding='utf8') as outj:
        json.dump(doc, outj, indent=2, ensure_ascii=False)

    with open(args.synout, 'w', encoding='utf8') as outs:
        json.dump(syn_groups, outs, indent=2, ensure_ascii=False)

    print(f"Wrote {len(annotations)} spans to {args.out}")
    print("Top terms by frequency:")
    for term, count in freq.most_common(20):
        print(f"  {term!r}: {count}")
    print(f"Synonym groups written to {args.synout}")

if __name__ == "__main__":
    main()
