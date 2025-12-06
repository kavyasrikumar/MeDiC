# Annotation Guidelines
These annotations will be stored in JSON files in the data/annotations folder!
## JSON Struct
See/copy the template in the data/templates folder! Naming should be doc123_annotation_v#.json
```
{
  "document_id": "doc123",
  "text": "Full text of the document here...",
  "annotations": [
    {
      "text_span_id": "span1",
      "start_offset": 102,
      "end_offset": 113,
      "annotated_text": "heart attack",
      "canonical_concept": "C0027051",
      "synonym_group_id": "syn_grp_001",
      "synonym_type": "exact",
      "notes": ""
    }
    // more annotations here...
  ]
}
```
## How to annotate!
1. Find the phrase you want to annotate that means the same medical thing
  e.g. "heart attack" <-> "myocardial infarction"
2. Highlight each phrase exactly as it appears (copy the exact words)
3. Link it to the standard med concept ID (like the code from MeSH or UMLS)
4. Label with type of synonym:
   - exact
   - abbrev
   - variant
   - related
   - misspelling
5. Record stand and end positions of the phrase in the text
Example

|Text Span|Start|End|Canonical Concept|Synonym Type|Notes|
|:---:|:---:|:---:|:---:|:---:|:---:|
|heart attack|102|113 | C0027051          | exact        |                     |
| myocardial infarction | 250   | 270 | C0027051          | exact        |                     |
| MI                    | 300   | 302 | C0027051          | abbreviation | Common abbreviation |
