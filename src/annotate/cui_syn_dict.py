import json
import pandas as pd

# -------------------------------------------------------
# Load UMLS synonyms from MRCONSO.RRF
# -------------------------------------------------------
def load_umls_synonyms(mrconso_path, out):
    print(f"Loading UMLS synonyms from {mrconso_path} ...")
    cui_to_synonyms = {}
    with open(mrconso_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f,1):
            fields = line.strip().split("|")
            if len(fields) < 15:
                continue
            cui = fields[0]
            str_term = fields[14].lower()
            if cui not in cui_to_synonyms:
                cui_to_synonyms[cui] = set()
            cui_to_synonyms[cui].add(str_term)
    for cui in cui_to_synonyms:
        cui_to_synonyms[cui] = sorted(list(cui_to_synonyms[cui]))
    with open(out, "w") as out_file:
        json.dump(cui_to_synonyms, out_file)
    print(f"Loaded {len(cui_to_synonyms)} normalized terms from UMLS.")

load_umls_synonyms("data/4_umls/MRCONSO.RRF", "src/annotate/cui_to_syn.json")