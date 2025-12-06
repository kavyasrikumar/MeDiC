import json
from collections import defaultdict

def load_medmentions(text_path, ann_path):
    docs = defaultdict(lambda: {"title": "", "abstract": "", "annotations": []})

    # Load MedMentions texts (title + abstract)
    with open(text_path, 'corpus_pubtator.txt', encoding='utf-8') as f:
        for line in f:
            pmid, part, text = line.strip().split("|", 2)
            if part == 't':
                docs[pmid]['title'] = text
            elif part == 'a':
                docs[pmid]['abstract'] = text

    # Load MedMentions annotations
    with open(ann_path, 'corpus_pubtator.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
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

# def load_umls_synonyms(mrconso_path):
#     cui_to_synonyms = defaultdict(set)
#     with open(mrconso_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             fields = line.strip().split('|')
#             if len(fields) < 15:
#                 continue
#             cui = fields[0]
#             term = fields[14].lower()
#             cui_to_synonyms[cui].add(term)
#     return cui_to_synonyms

# def build_synonym_groups(docs, cui_to_synonyms):
#     synonym_groups = {}
#     group_counter = 1

#     for doc in docs.values():
#         for ann in doc['annotations']:
#             cui = ann['cui']
#             if cui and cui not in synonym_groups:
#                 synonyms = sorted(cui_to_synonyms.get(cui, []))
#                 if not synonyms:
#                     synonyms = [ann['text'].lower()]
#                 group_id = f"syn_{group_counter:04d}"
#                 synonym_groups[cui] = {
#                     "synonym_group_id": group_id,
#                     "canonical_concept": cui,
#                     "synonyms": synonyms
#                 }
#                 group_counter += 1
#     return synonym_groups

def save_json_per_doc(docs, synonym_groups, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for pmid, doc in docs.items():
        # # Attach synonym group IDs for each annotation
        # for ann in doc['annotations']:
        #     cui = ann['cui']
        #     if cui in synonym_groups:
        #         ann['synonym_group_id'] = synonym_groups[cui]['synonym_group_id']
        #     else:
        #         ann['synonym_group_id'] = None
        out_path = os.path.join(output_dir, f"{pmid}.json")
        json.dump({
            "document_id": pmid,
            "text": doc['text'],
            "annotations": doc['annotations']
        }, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved {len(docs)} documents to {output_dir}")

def save_synonym_groups(synonym_groups, out_path):
    # Save synonym groups indexed by canonical_concept (CUI)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(synonym_groups, f, indent=2, ensure_ascii=False)
    print(f"Saved synonym groups to {out_path}")

if __name__ == "__main__":
    # Example usage -- update paths here
    medmentions_text_path = "MedMentions/st21pv_title_abstract.txt"
    medmentions_ann_path = "MedMentions/st21pv_pubtator_ann.txt"
    mrconso_path = "umls/MRCONSO.RRF"
    output_dir = "MM_MRCONSO_output_docs"
    synonym_out = "output_synonym_groups.json"

    print("Loading MedMentions...")
    docs = load_medmentions(medmentions_text_path, medmentions_ann_path)

    print("Loading UMLS synonyms...")
    cui_to_synonyms = load_umls_synonyms(mrconso_path)

    print("Building synonym groups...")
    synonym_groups = build_synonym_groups(docs, cui_to_synonyms)

    print("Saving JSON per document...")
    save_json_per_doc(docs, synonym_groups, output_dir)

    print("Saving synonym groups file...")
    save_synonym_groups(synonym_groups, synonym_out)
