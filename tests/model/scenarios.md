0: training process sanity check:
  1.  Data leakage checks:train/test split does not include exact duplicates or near-duplicates across files.
I. testing whether model satifies some minimum threshold of pre-defined evaluation metrics (Per-class minimum recall tests):
  1. OOD: MUST
    A. test input with model which obviously map to certain category out of 10 per each length bucket without using the name of that category (configs needed: 1. multiple generated texts per each of 10 categories, maybe differentiate by level; 2. clear categorization; 3. metrics code definitions) PRIORITY
    B. test multiple variants of "other" category making sure it is not anyhow related to any of 10 categories (configs needed: 1. multiple generated texts per "other" category, maybe differentiate by level; 2. clear categorization; 3. metrics code definitions) PRIORITY
  2. ID tests: MUST
    - testing on training data
  
II. testing minimum pure model inference latency on generated data test cases from "I." point above (from lowest to highest). MUST

III. testing throughput 

IV. change of expectation testing:
 1. identify some training input from 1 out of 10 classes for which it is easier to convert it into some other class and sent to api to see if prediction changed. MUST
 2. mixed input between two classes (the only question is what do we test in such a way, haha)
 3. Class keyword trap tests: text contains a strong keyword from one class but actual context belongs to another, e.g. “the campaign raised startup funding” could confuse politics/business. MUST
 4. Long mixed document tests: mostly one class with a small paragraph from another class; define whether majority topic should win.
 5. very short ambiguous input: one sentence or headline-like text with limited evidence; useful because TF-IDF may overtrust rare words.
 6. Near-duplicate regression tests: paraphrases of known examples should keep the same label, while small semantic edits should change label.
 7. Vocabulary drift tests: modern terms not seen in training, e.g. new tech/product names, new medical terminology, current political phrases.
 8. Top-2 margin tests for intentionally mixed inputs between two known classes, verify that both intended classes appear in
   the top-2 predictions and their probability gap is not too large. MUST
 9. confidence threshold tests with different "other" OOD examples to ensure we are mostly returning "other" instead of known class MUST 

V. invariance testing:
  1. typos MUST
  2. lowercase MUST
  3. punctuation MUST
  4. increased/decreased ID and OOD data to test if we are able to map to the same class
  5. boilerplate/noise robustness: add headers, signatures, URLs, emails, markdown, HTML fragments, logs, or quoted text before/after the main content.
  6. formatting invariance: newlines, tabs, bullet lists, repeated spaces, ALL CAPS, unicode punctuation, numbers, and abbreviations.
  7. stopword/negation sensitivity: “not a sports article”, “not medical advice”, “politics-free business update”; TF-IDF may ignore intent if keywords dominate. MUST
