# Exercise 02 — Text Classifier

## Overview
This exercise builds a **text classifier** that uses an LLM (Groq) to categorize short documents into predefined categories and sub-categories, returning strictly valid JSON.

## Prompting Techniques Implemented
| # | Technique | Description |
|---|-----------|-------------|
| 1 | Zero-shot | No examples — relies entirely on the model's pre-trained knowledge |
| 2 | Few-shot | 2 examples provided before the actual text |
| 3 | Chain-of-Thought | Step-by-step internal reasoning before producing the output |

## Classification Strategy — Option B (2-Call Chain)
To **minimize hallucination risk**, classification is split into two sequential LLM calls:
1. **Call 1 → Category** — The model picks from the 5 allowed categories.
2. **Call 2 → Sub-category** — The model picks from sub-categories *constrained* to the chosen category.

## Allowed Taxonomy
| Category | Sub-categories |
|----------|---------------|
| Reviews | Positive, Negative, Neutral |
| Marketing & Promotion | Product Promotion, Service Promotion, Discount - Offer, Brand Awareness |
| Academic | History, Ethics, Business, Geography |
| Sports/Leagues | American Football / NFL, Basketball / MLB, Soccer / MLS, Hockey / NHL, Women Basketball / WNBA |
| Finance & Expenses | Personal expenses, Business expenses, Travel expenses, Budget planning |

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Groq](https://console.groq.com/) API key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
copy .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 4. Run the classifier
```bash
python classifier.py
```
Select prompting technique (1 / 2 / 3) when prompted. The script will:
- Load all 32 `dataset/*.txt` files
- Classify each document using a 2-call LLM chain
- Print live results to the console
- Save all results to `results.json`

## Output Format
```json
{
  "document_name": "review_positive_01.txt",
  "category": "Reviews",
  "sub_category": "Positive"
}
```

## File Structure
```
Exercise_02/
├── classifier.py       ← Main script
├── requirements.txt
├── .env.example
├── README.md           ← This file
├── dataset/            ← 32 sample text documents
│   ├── review_positive_01.txt
│   ├── ...
└── results.json        ← Generated after running the classifier
```
