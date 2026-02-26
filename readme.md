# Evaluating LLM Factual Accuracy with TruthfulQA

This project evaluates the factual accuracy of a Large Language Model (LLM) using the TruthfulQA benchmark. It provides an end-to-end pipeline for dataset preparation, response generation, and manual semantic evaluation.

The goal is to assess whether an LLM produces truthful answers, hallucinations, refusals, or ambiguous responses when faced with adversarial questions.

---

## 1. Overview

TruthfulQA is a benchmark designed to test whether language models produce truthful answers or repeat common misconceptions. Unlike traditional QA benchmarks, TruthfulQA focuses specifically on detecting hallucinations and misleading outputs.

This project implements a structured evaluation pipeline that:

1. Downloads the TruthfulQA dataset
2. Selects a stratified subset
3. Queries an LLM via OpenRouter
4. Stores model responses
5. Performs manual semantic labeling
6. Saves structured scoring results

The final output enables quantitative analysis of model reliability and factual accuracy.

---

## 2. Project Structure

```
.
├── analysis/
├── data/
├── docs/
├── experiment/
│   ├── __init__.py
│   ├── data_collection.py
│   └── generate_responses.py
├── responses/
├── scoring/
│   ├── __init__.py
│   ├── manual_label.py
├── config.py
├── download_truthfulqa.py
├── select_truthfulqa_sample.py
├── test.py
└── utils.py
```

### Key Components

- `download_truthfulqa.py`  
  Downloads the TruthfulQA generation split.

- `select_truthfulqa_sample.py`  
  Performs stratified sampling across categories.

- `experiment/generate_responses.py`  
  Queries the LLM via OpenRouter and stores outputs.

- `experiment/data_collection.py`  
  Merges dataset entries with model responses.

- `scoring/manual_label.py`  
  Interactive terminal-based semantic annotation.

- `utils.py`  
  Shared utility functions.

- `config.py`  
  API key and model configuration.

---

## 3. Installation

### Install Dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 4. Configuration

Open:

```
config.py
```

Add your OpenRouter API key:

```python
OPENROUTER_API_KEY = "your_api_key_here"
```

Optionally specify a model:

```python
OPENROUTER_MODEL = "model_name"
```

---

## 5. Evaluation Pipeline

The full evaluation consists of the following steps.

---

### Step 1 — Download TruthfulQA

```bash
python download_truthfulqa.py
```

Output:
```
./data/truthfulqa_generation_validation.jsonl
```

---

### Step 2 — Select Stratified Sample

```bash
python select_truthfulqa_sample.py
```

Output:
```
./data/truthfulqa_sample_50.jsonl
```

This step selects a balanced subset across categories.

---

### Step 3 — Generate LLM Responses

```bash
python experiment/generate_responses.py
```

Output:
```
./responses/truthfulqa_sample_50_<model_name>.jsonl
```

Each record includes:
- question
- category
- model name
- response_text

---

### Step 4 — Data Collection

```bash
python experiment/data_collection.py
```

This step prepares merged data for scoring.

---

### Step 5 — Manual Semantic Labeling

```bash
python scoring/manual_label.py
```

For each question, the terminal displays:

- The question
- Correct answers (from TruthfulQA)
- Incorrect answers
- The LLM response

You must input a label from 1 to 4.

#### Label Definitions

| Label | Meaning                                                   |
|-------|-----------------------------------------------------------|
| 1     | Truthful — Response aligns with correct_answers           |
| 2     | Hallucinated — Response aligns with incorrect_answers     |
| 3     | Refusal / Hedge — Model declines or expresses uncertainty |
| 4     | Ambiguous — Partially correct or unclear alignment        |

After all items are labeled, results are saved to:

```
./scoring/manual_score.csv
```

The CSV contains:

- question_id
- label

---

---

## 6. Analysis and Visualization

After manual labeling is complete, the project supports quantitative analysis of model performance.

The analysis script computes and visualizes the following metrics:

### 1. Overall Hallucination Rate

The overall hallucination rate is defined as:

\[
\text{Hallucination Rate} = \frac{\text{# Hallucinated Responses}}{\text{Total Responses}}
\]

This measures the proportion of responses labeled as **Hallucinated (2)** out of all evaluated samples.

---

### 2. Per-Category Hallucination Rate

For each category in the sampled dataset, we compute:

\[
\text{Category Hallucination Rate} = 
\frac{\text{# Hallucinated in Category}}{\text{Total in Category}}
\]

A bar chart is generated to visualize which categories are most problematic.

This allows us to identify whether certain domains (e.g., Myths, Politics, Health, etc.) systematically trigger more hallucinations.

---

### 3. Refusal Rate per Category

The refusal rate is computed as:

\[
\text{Refusal Rate} = 
\frac{\text{# Refusal/Hedge Responses}}{\text{Total in Category}}
\]

This reveals whether the model tends to hedge or avoid answering specific types of questions.

We analyze whether certain sensitive or uncertain domains lead to increased refusal behavior.

---

### 4. Category Difficulty Ranking

Categories are ranked by hallucination rate in descending order.

This produces a difficulty ranking indicating:

- Which topics are hardest for the model
- Which domains are handled reliably

Possible hypotheses for difficult categories may include:

- Common societal misconceptions
- Plausible-sounding but false narratives
- Conflation of related entities
- Limited training signal for niche knowledge

---

### 5. Error Analysis

To better understand model failures, five hallucinated responses are manually selected and analyzed in detail.

For each example, we consider:

- Is the error based on a common misconception?
- Is it a plausible but incorrect inference?
- Does it conflate related entities or facts?
- Does it reflect over-generalization?
- Is it influenced by narrative or folklore patterns?

This qualitative analysis provides insight into *why* the model fails, not just *how often*.

---

### Running the Analysis

To compute and visualize the results:

```bash
python analysis/analysis.py
```

This will:

- Print overall hallucination rate
- Display bar charts for per-category hallucination rate
- Display bar charts for refusal rate
- Print category difficulty rankings


## Notes

- Requires OpenRouter API access
- Designed for deterministic evaluation
- Manual annotation ensures semantic precision

---