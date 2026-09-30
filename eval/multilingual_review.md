# Multilingual review checklist

Use this after a real V2 benchmark run. This is a human QA checklist, not a scored result.

For a sample of English, Hausa, Yoruba, and Igbo answers, verify:

- the detected language matches the question;
- the answer is in that same language;
- the selected tool matches the user's intent rather than literal keywords;
- knowledge questions use an English retrieval query while preserving the original user question for the final answer;
- URLs are preserved exactly;
- unsupported questions do not invent rules, contacts, dates, or benefits;
- policy answers are traceable to the retrieved official source passages;
- equivalent questions across languages select the same tool unless their wording genuinely changes the intent.

Record any systematic language-specific routing or fluency issue next to the benchmark result file that exposed it.
