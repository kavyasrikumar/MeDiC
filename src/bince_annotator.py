# pip install spacy scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

import json
import uuid
import spacy
import sys
from scispacy.linking import EntityLinker

print("Script started!")

# load model and entity linker
print("Loading SciSpaCy model…")

# A full spaCy pipeline for biomedical data with a ~785k vocabulary and 600k word vectors.
nlp = spacy.load("en_core_sci_lg-0.5.4.tar.gz")  

print("Loading entity linker…")
linker = EntityLinker(resolve_abbreviations=True, max_entities_per_mention=1)
nlp.add_pipe("scispacy_linker", config={"linker": linker})

# annotation object builder : technically just a skeleton bc it doesn't actually work at the moment
def build_annotation(span, cui):
    return {
        "text_span_id": "span_" + str(uuid.uuid4())[:8],
        "start_offset": span.start_char,
        "end_offset": span.end_char,
        "annotated_text": span.text,
        "canonical_concept": cui,
        "synonym_group_id": f"syn_{cui.lower()}",
        "synonym_type": "preferred",
        "notes": "automatic umls annotation"
    }

# Driver Code
if __name__ == "__main__":
    with open(sys.argv[1]) as files:
        chunk = json.load(files)
        print(chunk)
    
    # TODO: print out / write into file?
    #result = annotate_text(input_text)

    #print(json.dumps(result, indent=2))
