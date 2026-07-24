# I.1.A Known-Class Length Fixture Prompt

Model: {{model}}
Prompt version: {{prompt_version}}
Known labels: {{known_labels}}
Target label: {{expected_label}}
Target bucket: {{length_bucket}}
Target word range: {{min_words}}-{{max_words}}
Prompt-level generation seed: {{generation_seed}}
Attempt: {{attempt}}

Write one original English document that clearly belongs to the target label.

Quality rules:
- Return only valid JSON in this exact shape: {"text": "..."}.
- Do not include markdown, explanations, arrays, or additional fields.
- Do not use the literal target label as a standalone word anywhere in the text.
- Keep the text within the target word range and comfortably inside the project length bucket.
- Make the subject matter specific enough that a text classifier should infer the target label.
- For ultra-short rows, pack the text with concrete domain terms rather than generic wording.
- Avoid mentioning other known labels or writing a meta description of the task.

Project tokenizer buckets:
- ultra_short: 0-19 tokens
- short: 20-126 tokens
- medium: 127-445 tokens
- long: 446-779 tokens
- extra_long: 780+ tokens
