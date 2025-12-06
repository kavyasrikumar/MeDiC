import json
from collections import defaultdict
import os
import zipfile

def ensure_mrconso_unzipped(zip_path, extract_dir="."):
    extracted_path = os.path.join(extract_dir, "MRCONSO.RRF")
    if not os.path.isfile(extracted_path):
        print(f"Extracting MRCONSO.RRF from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("MRCONSO.RRF", path=extract_dir)
        print("Extraction complete.")
    else:
        print("MRCONSO.RRF already extracted.")
    return extracted_path

def load_medmentions(text_path):
    docs = defaultdict(lambda: {"title": "", "abstract": "", "annotations": []})

    # Load MedMentions texts (title + abstract and annotations)
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                pmid, part, text = parts[0], parts[1], parts[2]
                if part == 't':
                    docs[pmid]['title'] = text
                elif part == 'a':
                    docs[pmid]['abstract'] = text
            elif len(parts) == 6 and parts[1].isnumeric():
                pmid, start, end, span_text, sem_types, cui = parts
                start, end = int(start), int(end)
                sem_types = sem_types.split(",")
                docs[pmid]['annotations'].append({
                    "start_offset": start,
                    "end_offset": end,
                    "text": span_text,
                    "semantic_types": sem_types,
                    "cui": cui
                })

    # Concatenate title + abstract
    for pmid, doc in docs.items():
        doc['text'] = doc['title'] + "\n" + doc['abstract']
        del doc['title']
        del doc['abstract']

    return docs

def load_umls_synonyms(mrconso_path, source_vocab_filter=None):
    cui_to_synonyms = defaultdict(set)
    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) < 15:
                continue
            cui = fields[0]
            sab = fields[11]
            term = fields[14].lower()
            if source_vocab_filter and sab not in source_vocab_filter:
                continue
            if term:
                cui_to_synonyms[cui].add(term)
    return cui_to_synonyms

def build_synonym_groups(docs, cui_to_synonyms):
    synonym_groups = {}
    group_counter = 1

    for doc in docs.values():
        for ann in doc['annotations']:
            cui = ann['cui']
            if cui and cui not in synonym_groups:
                synonyms = sorted(cui_to_synonyms.get(cui, []))
                if not synonyms:
                    synonyms = [ann['text'].lower()]
                group_id = f"syn_{group_counter:04d}"
                synonym_groups[cui] = {
                    "synonym_group_id": group_id,
                    "canonical_concept": cui,
                    "synonyms": synonyms
                }
                group_counter += 1
    return synonym_groups

def save_json_per_doc(docs, synonym_groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pmid, doc in docs.items():
        # Attach synonym group IDs for each annotation
        for ann in doc['annotations']:
            cui = ann['cui']
            if cui in synonym_groups:
                ann['synonym_group_id'] = synonym_groups[cui]['synonym_group_id']
            else:
                ann['synonym_group_id'] = None
        out_path = os.path.join(output_dir, f"{pmid}.json")
        json.dump({
            "document_id": pmid,
            "text": doc['text'],
            "annotations": doc['annotations']
        }, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved {len(docs)} documents to {output_dir}")

def save_synonym_groups(synonym_groups, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(synonym_groups, f, indent=2, ensure_ascii=False)
    print(f"Saved synonym groups to {out_path}")

if __name__ == "__main__":
    medmentions_text_path = "corpus_pubtator.txt"  # Your MedMentions file path
    mrconso_zip_path = "MRCONSO.RRF.zip"           # Your zipped MRCONSO file path
    extract_dir = "data"                           # Extraction directory
    output_dir = "MM_MRCONSO_output_docs"          # Output folder for docs JSON
    synonym_out = "output_synonym_groups.json"     # Output file for synonym groups

    print("Loading MedMentions...")
    docs = load_medmentions(medmentions_text_path)

    print("Extracting MRCONSO if needed...")
    mrconso_path = ensure_mrconso_unzipped(mrconso_zip_path, extract_dir)

    print("Loading UMLS synonyms...")
    # Filter by source vocabularies you care about (adjust as needed)
    source_vocab_filter = ['SNOMEDCT_US', 'ICD10CM', 'RXNORM', 'LNC'] 
    cui_to_synonyms = load_umls_synonyms(mrconso_path, source_vocab_filter)

    print("Building synonym groups...")
    synonym_groups = build_synonym_groups(docs, cui_to_synonyms)

    print("Saving JSON per document...")
    save_json_per_doc(docs, synonym_groups, output_dir)

    print("Saving synonym groups file...")
    save_synonym_groups(synonym_groups, synonym_out)