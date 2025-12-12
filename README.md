# Medical-Disambiguation-in-Context

## Repo Struct

```
data/
  raw/            # All the manually annotated work; still need to compile into one GOLD file for testing
  processed/      # All the "processed" annotated data from MedMention, and will be split into dev and training
    like 4000+ MedMention annotated .json files
    manual_ann_docs.jsonl   # This will store the data from the manual annotations as one usable .jsonl file for GOLD
  synonym_groups.json       # Our gorgeous master list of synonyms per CUI parsed
src/
  annotate/       # All the annotators and necessary scripts 
    cui_syn_dict.py # NO NEED TO RUN AGAIN (was used to generate the synonym groups json for easier parsing)
    cui_to_syn.json # CUI number mappings to synonyms; used to inform the autoannotator and also can be used as source of truth for synonym embeddings with CUIs
    data_web        # This is only to generate new raw files for manual annotation, I don't see us needing to run this again
    interactive.py  # As the name suggests, this is just for assisted annotations
  preprocess/     # ONLY NEED RERUN UMLS_LOADER_NORMALIZE.py to update the synonym groups post annotation update
    __pycache__         # dw about this, it's just cache
    MM_to_processed.py  # used to load MedMention annotations into the format we wanted to use in .jsons; use this to see how many documents we actually used (it prints a line at the end for that)
    umls_loader_norm.py # This is the biggest driver: it ultimately creates a master synonym_group.json that combines UMLS CUI synonyms, the MedMention synonyms, and our manually annotated synonyms AND the manual_ann_docs.json
.gitattributes: we have this storing the big files like MRCONSO.RRF and the bajillion documents
corpus_pubtator.txt     # the MedMention Dataset
MRCONSO.RRF             # UMLS Dataset used (don't try to open it this is massive af)
```
## Annotation Workflow
- Where annotation templates are
- [Annotation Guidelines](https://github.com/kavyasrikumar/MeDiC/blob/3e40638dd9202906ba9634a48530cb154b20f688/data/annotation_README.md)

## Model Details
- We used SapBERT-from-PubMedBERT-fulltext (Liu et al., 2021) as it is designed for biomedical concept normalization through its synonym-aware and domain-specific capabilities.

## Evaluation
- To perform evaluation the following steps were taken:
  - A development set of 50 documents were used for threshold tuning and hyperparameter selection
  - MeDiC pipeline was run: candidate spans were extracted using SciSpacy NER
  - Each candidate span was encoded using SapBERT and its concept embeddings were compared with a filtered applied to it using similarity thresholds and confidence-gap thresholds.
- After the pipeline was run the best configuration used:
  - a similarity threshold of 0.75
  - a confidence gap of 0.05
  
## Results
- The Results showcased:
  - a precision of 68%
  - a recall score of 46.5%
  - an F1 score of 51.4%
