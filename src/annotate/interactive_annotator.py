import json
import re
import argparse
import os

# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_json(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_umls_synonyms(path):
    """
    Load a preprocessed smaller UMLS synonyms file.
    Expected format:
      { "tuberculosis": [ {"cui": "C0041296", "preferred": "..."} ] }
    """
    print(f"Loading UMLS synonyms from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        umls = json.load(f)
    print(f"Loaded {len(umls)} synonym keys.")
    return umls

# ------------------------------
# Annotation helpers
# ------------------------------

def generate_span_id(counter):
    return f"span_{counter}"


def handle_user_selection(selected_text, umls_dict):
    """
    Show suggestions IF available, but no auto-annotation.
    Returns dict for annotation fields or None.
    """
    print(f"\nSelected text: '{selected_text}'")

    norm = selected_text.lower()
    suggestions = umls_dict.get(norm, [])

    if suggestions:
        print("\n🔎 CUI Suggestions:")
        for s in suggestions:
            print(f"  • CUI: {s['cui']} | Pref: {s.get('preferred','')} | Syn: {s.get('synonym','')}")
    else:
        print("No CUI suggestions found.")

    canonical_cui = input("\nEnter canonical CUI (or ENTER to skip): ").strip()
    if not canonical_cui:
        print("Skipping annotation.\n")
        return None

    syn_group = input("Synonym group ID (e.g., syn_tb): ").strip()
    syn_type = input("Synonym type (preferred / abbreviation / alt): ").strip()
    notes = input("Notes: ").strip()

    return {
        "canonical_cui": canonical_cui,
        "syn_group": syn_group,
        "syn_type": syn_type,
        "notes": notes
    }


def annotate_all_occurrences(text, selected_text, annotation_data, start_counter):
    """
    Annotates ALL occurrences of selected_text in the article.
    """
    annotations = []
    pattern = re.escape(selected_text)
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    print(f"\nFound {len(matches)} occurrences.")

    for m in matches:
        span_id = generate_span_id(start_counter)

        annotations.append({
            "text_span_id": span_id,
            "start_offset": m.start(),
            "end_offset": m.end(),
            "annotated_text": m.group(0),
            "canonical_concept": annotation_data["canonical_cui"],
            "synonym_group_id": annotation_data["syn_group"],
            "synonym_type": annotation_data["syn_type"],
            "notes": annotation_data["notes"]
        })

        start_counter += 1

    return annotations, start_counter

# ------------------------------
# MAIN
# ------------------------------

def main():
    print("\n=== Interactive Annotator ===")

    # -------------------------
    # File inputs
    # -------------------------
    raw_path = input("Enter path to raw JSONL file: ").strip()
    while not os.path.exists(raw_path):
        print("File not found. Try again.")
        raw_path = input("Enter path to raw JSONL file: ").strip()

    umls_path = input("Enter path to UMLS synonym JSON: ").strip()
    while not os.path.exists(umls_path):
        print("File not found. Try again.")
        umls_path = input("Enter path to UMLS synonym JSON: ").strip()

    data = load_json(raw_path)
    umls_dict = load_umls_synonyms(umls_path)

    # Assume raw JSONL structure:
    # { "doc_id": "...", "url": "...", "text": "full text" }
    print(f"Loaded {len(data)} documents.\n")

    annotated_items = []
    span_counter = 1

    text = data["text"]
    annotations = []

    while True:
        print("\n--- Annotation Menu ---")
        print("1. Annotate text")
        print("2. View current annotations")
        print("3. Done with this document")
        choice = input("> ").strip()

        if choice == "1":
            selected = input("\nEnter EXACT text to annotate: ").strip()
            if not selected:
                print("No text entered.")
                continue

            ann_info = handle_user_selection(selected, umls_dict)
            if ann_info:
                new_anns, span_counter = annotate_all_occurrences(
                    text, selected, ann_info, span_counter
                )
                annotations.extend(new_anns)
                print(f"Added {len(new_anns)} annotations.")

        elif choice == "2":
            print("\nCurrent annotations:")
            for ann in annotations:
                print(json.dumps(ann, indent=2))
        else:
            break

    data["annotations"] = annotations
    annotated_items.append(data)

    # -------------------------
    # Save output
    # -------------------------
    out_path = "data/2_processed/manual_annotations.jsonl"
    save_jsonl(out_path, annotated_items)
    print(f"\nSaved annotations to {out_path}")


if __name__ == "__main__":
    main()