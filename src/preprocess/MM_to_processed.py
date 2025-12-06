from collections import defaultdict

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

def main():
    docs = load_medmentions_pubtator("corpus_pubtator.txt")
    print(f"[INFO] Loaded {len(docs)} documents from corpus_pubtator.txt")