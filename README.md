# ⚡ BERT-GPT Hybrid QA Engine

> A hybrid Natural Language Processing pipeline combining **BERT**'s extractive precision with **GPT**'s generative capability for accurate, context-grounded text synthesis.
> 
## 📌 Project Overview

This project bridges two distinct transformer architectures to deliver high-accuracy Question Answering and text generation:

1. **BERT (Extractive Phase):** Scans dense context text to identify and extract the precise answer span with zero hallucination.
2. **GPT (Generative Phase):** Receives the extracted fact alongside the prompt to autoregressively generate fluid, detailed continuations and downstream explanations.

By pre-filtering long context through BERT before passing key facts to GPT, the system reduces token bloat and keeps generative outputs strictly grounded in truth.

## 🏗️ Architecture Flow

[ Raw Context + Question ] 
            │
            ▼
┌───────────────────────────┐
│     BERT (Extractor)      │  ──► Pinpoints exact answer span
└───────────────────────────┘
            │
            ▼  (Extracted Baseline Fact)
┌───────────────────────────┐
│      GPT (Generator)      │  ──► Generates rich, continuous response
└───────────────────────────┘
            │
            ▼
   [ Final Output ]
