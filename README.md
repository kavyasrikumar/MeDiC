# Medical-Disambiguation-in-Context

## Repo Struct

```
data/
  kaggle_splits   # This is where the dev/train/test files live for our local pushes, but they are accessible 
  processed/      # All the "processed" annotated data from MedMention, and will be split into dev and training
    MM_docs                  # 4000+ MedMention annotated .json files
    comb_and_split.py        # NO NEED TO RUN AGAIN (was used to generate data splits from processed data
    manual_ann_docs.jsonl    # This will store the data from the manual annotations as one usable .jsonl file for GOLD
  synonym_groups.json    # Our gorgeous master list of synonyms per CUI parsed
evaluation/  
  results/        # final result data used to drive fine-tuning and threshold adjustments
    full_dataset_statistics.json
    scispacy_ner_test.txt
    threshold_tuning_results.json
  analyze_missed_entities.py
  error_analysis.py
  evaluate_sapbert.py
  full_statistics.py
  run_evaluation.py
  statistical_analysis.py
  tune_sapbery_threshold.py
models/
  sapbert_embeddings.json
  sapbert_matcher.py
src/
  annotate/       # All the annotators and necessary scripts
    annotation_README.md      # annotation guidelines for standardization
    cui_syn_dict.py           # NO NEED TO RUN AGAIN (was used to generate the synonym groups json for easier parsing)
    cui_to_syn.json           # CUI number mappings to synonyms; used to inform the autoannotator and also can be used as source of truth for synonym embeddings with CUIs
    data_web_extract.py       # This is only to generate new raw files for manual annotation, I don't see us needing to run this again
    interactive_annotator.py  # As the name suggests, this is just for assisted annotations
  preprocess/     # ONLY NEED RERUN UMLS_LOADER_NORMALIZE.py to update the synonym groups post annotation update
    MM_to_processed.py        # used to load MedMention annotations into the format we wanted to use in .jsons; use this to see how many documents we actually used (it prints a line at the end for that)
    umls_loader_norm.py       # This is the biggest driver: it ultimately creates a master synonym_group.json that combines UMLS CUI synonyms, the MedMention synonyms, and our manually annotated synonyms AND the manual_ann_docs.json
  

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
