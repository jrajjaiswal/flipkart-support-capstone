# Flipkart Support Capstone

## Project Overview

This repository contains the three-part Flipkart Support Capstone:

- Part 1 — Return-risk prediction
- Part 2 — Product-category image classification
- Part 3 — Retrieval-augmented support agent using LangGraph

## Part 1 — Return Risk

Main files:

- `generate_orders.py`
- `orders_dataset.csv`
- `train_return_risk.py`
- `models/return_risk_model.pkl`

The saved Random Forest is used by the Part 3 `check_return_risk` tool.

Risk threshold:

- `t*_rf = 0.47`
- Low: probability < 0.47
- Medium: 0.47 to < 0.62
- High: probability >= 0.62

## Part 2 — Product Image Classification

Main files:

- `train_product_classifier.py`
- `models/product_classifier.pt`
- `data/sample_images/`

Verified sample:

`00001_Pullover.png`

Result:

- Predicted category: `Pullover`
- Confidence: approximately `0.9706`

## Part 3 — Flipkart Support Agent

Main files:

- `part3_agent.py`
- `policy_knowledge_base.json`
- `build_retrieval_index.py`
- `retrieval_evaluation.json`
- `transcripts/`

### Knowledge Base

- 12 policy documents
- 21 sentence-wise chunks

### Retrieval

- Embedding model: `all-MiniLM-L6-v2`
- Vector index: `Faiss IndexFlatIP`
- Top-k retrieval: 3

### LangGraph

Application nodes:

1. `input_guard`
2. `intent`
3. `policy`
4. `risk`
5. `image`
6. `response`
7. `output_guard`

The graph uses conditional routing.

### Intents

- `policy`
- `return_risk`
- `product_image`
- `general`

### Real Tools

`check_return_risk` loads:

`models/return_risk_model.pkl`

`classify_product_image` loads:

`models/product_classifier.pt`

and operates on real PNG files in `data/sample_images/`.

### Guardrails

Input-side prompt-injection filtering blocks patterns such as:

- `ignore previous instructions`
- `ignore all rules`
- `reveal your system prompt`

Output-side groundedness threshold:

`0.35`

A policy question below this threshold is refused rather than fabricated.

### MOCK_LLM

The graded mode is deterministic `MOCK_LLM`.

It requires zero API keys and zero outbound LLM calls.

### Prompt Design

The response-generation configuration includes:

- Role prompting
- 4S — Specific
- 4S — Short
- 4S — Surround
- 4S — Single
- Few-shot intent examples

## Test Transcripts

The required transcripts are stored in `transcripts/`.

- [T01 — Apparel policy](transcripts/T01_policy_apparel.json)
- [T02 — COD refund policy](transcripts/T02_policy_cod_refund.json)
- [T03 — Return-risk tool](transcripts/T03_return_risk.json)
- [T04 — Product-image tool](transcripts/T04_product_image.json)
- [T05 — Multi-turn state](transcripts/T05_multiturn_state.json)
- [T05 — Fresh conversation](transcripts/T05_fresh_conversation.json)
- [T06 — Prompt injection](transcripts/T06_prompt_injection.json)
- [T07 — Ungrounded policy refusal](transcripts/T07_ungrounded_policy.json)
- [T08 — General conversation](transcripts/T08_general.json)

The multi-turn transcript demonstrates state retained within one conversation, while the separate fresh-conversation transcript starts with no previous messages.

## Retrieval Evaluation

Evaluation is performed at the parent-document level after mapping chunks to parent documents and deduplicating them.

Results:

- Average Precision@3: `0.4333`
- Average Recall@3: `0.9000`

Full evaluation:

[retrieval_evaluation.json](retrieval_evaluation.json)

## Verified Behaviors

Verified behaviors include:

- Policy retrieval
- Return-risk prediction
- Product-image classification
- Prompt-injection blocking
- Groundedness refusal
- Multi-turn state
- Fresh-conversation reset
- Structured response output

Verified return-risk result:

- Probability: `0.61884219`
- Risk bucket: `Medium`

Verified ungrounded-policy case:

- Maximum similarity: `0.2646`
- Threshold: `0.35`
- Result: refusal

## Basic Run Order

### Part 1

- `generate_orders.py`
- `train_return_risk.py`

### Part 2

- `train_product_classifier.py`

### Part 3

- `build_retrieval_index.py`
- `part3_agent.py`

The Part 3 implementation rebuilds the knowledge-base embeddings and Faiss index and loads the saved Part 1 and Part 2 model artifacts.

## Submission Contents

The repository contains the Part 1 and Part 2 models, Part 3 agent source, knowledge base, retrieval build code, retrieval evaluation, transcripts, and this root README.
