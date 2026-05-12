"""
GEN AI Upskilling Training — Exercise 02
Text Classifier with Structured JSON Output using Groq

Prompting techniques: Zero-shot, Few-shot, Chain-of-Thought
Classification strategy: Option B (2-call chain) to reduce hallucination risk
  Call 1 → Determine category
  Call 2 → Determine sub-category (constrained by category from Call 1)
"""

import os
import json
import glob
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────
# TAXONOMY — Allowed categories and their sub-categories
# ─────────────────────────────────────────────────────────────
TAXONOMY = {
    "Reviews": ["Positive", "Negative", "Neutral"],
    "Marketing & Promotion": [
        "Product Promotion",
        "Service Promotion",
        "Discount - Offer",
        "Brand Awareness",
    ],
    "Academic": ["History", "Ethics", "Business", "Geography"],
    "Sports/Leagues": [
        "American Football / NFL",
        "Basketball / MLB",
        "Soccer / MLS",
        "Hockey / NHL",
        "Women Basketball / WNBA",
    ],
    "Finance & Expenses": [
        "Personal expenses",
        "Business expenses",
        "Travel expenses",
        "Budget planning",
    ],
}

CATEGORIES = list(TAXONOMY.keys())

# ─────────────────────────────────────────────────────────────
# STEP 1 — PROMPT DEFINITIONS (3 techniques)
#           Call 1: Determine CATEGORY
# ─────────────────────────────────────────────────────────────

CATEGORY_PROMPT_ZEROSHOT = f"""
You are an expert text classification assistant.
Your task is to read a text and determine its category.

Allowed categories (choose EXACTLY one):
{json.dumps(CATEGORIES, indent=2)}

IMPORTANT: Respond ONLY with a valid JSON object. No markdown, no explanation.

Required format:
{{"category": "<one of the allowed categories>"}}
""".strip()

CATEGORY_PROMPT_FEWSHOT = f"""
You are an expert text classification assistant.
Your task is to read a text and determine its category.

Allowed categories (choose EXACTLY one):
{json.dumps(CATEGORIES, indent=2)}

Here are two examples:

Example 1:
Text: "I loved this blender! It blends smoothies perfectly and is very easy to clean."
Output: {{"category": "Reviews"}}

Example 2:
Text: "The team scored three touchdowns in the final quarter to win the NFL playoff game."
Output: {{"category": "Sports/Leagues"}}

Now apply the same logic to the user's text. Respond ONLY with valid JSON.
""".strip()

CATEGORY_PROMPT_COT = f"""
You are an expert text classification assistant.

Follow these internal reasoning steps:
  Step 1: Read the text carefully and identify the main topic.
  Step 2: Consider which of the allowed categories best fits the topic.
  Step 3: Eliminate categories that do not match.
  Step 4: Select the single best category.

Allowed categories:
{json.dumps(CATEGORIES, indent=2)}

After completing your reasoning, respond ONLY with the following valid JSON.
Do NOT include reasoning steps in your response.

Required format:
{{"category": "<one of the allowed categories>"}}
""".strip()

# ─────────────────────────────────────────────────────────────
# STEP 2 — PROMPT DEFINITIONS (3 techniques)
#           Call 2: Determine SUB-CATEGORY (constrained)
# ─────────────────────────────────────────────────────────────

def build_subcategory_prompt_zeroshot(category: str) -> str:
    subcategories = TAXONOMY[category]
    return f"""
You are an expert text classification assistant.
The text has already been classified under the category: "{category}".

Your task is to determine the most appropriate sub-category.

Allowed sub-categories for "{category}" (choose EXACTLY one):
{json.dumps(subcategories, indent=2)}

IMPORTANT: Respond ONLY with a valid JSON object. No markdown, no explanation.

Required format:
{{"sub_category": "<one of the allowed sub-categories>"}}
""".strip()


def build_subcategory_prompt_fewshot(category: str) -> str:
    subcategories = TAXONOMY[category]
    return f"""
You are an expert text classification assistant.
The text has already been classified under the category: "{category}".

Your task is to determine the most appropriate sub-category.

Allowed sub-categories for "{category}":
{json.dumps(subcategories, indent=2)}

Here are two examples:
Example 1 (category=Reviews):
Text: "The product was okay. Not great, not terrible."
Output: {{"sub_category": "Neutral"}}

Example 2 (category=Finance & Expenses):
Text: "I tracked every dollar I spent on food, rent, and clothing this month."
Output: {{"sub_category": "Personal expenses"}}

Now apply the same logic to the user's text. Respond ONLY with valid JSON.
""".strip()


def build_subcategory_prompt_cot(category: str) -> str:
    subcategories = TAXONOMY[category]
    return f"""
You are an expert text classification assistant.
The text has already been classified under the category: "{category}".

Follow these reasoning steps:
  Step 1: Re-read the text focusing on nuances related to "{category}".
  Step 2: Compare the text against each allowed sub-category.
  Step 3: Select the single best-matching sub-category.

Allowed sub-categories for "{category}":
{json.dumps(subcategories, indent=2)}

After completing your reasoning, respond ONLY with this valid JSON.
Do NOT include reasoning in the response.

Required format:
{{"sub_category": "<one of the allowed sub-categories>"}}
""".strip()


PROMPTS = {
    "1": {
        "name": "Zero-shot",
        "category_prompt": CATEGORY_PROMPT_ZEROSHOT,
        "subcategory_builder": build_subcategory_prompt_zeroshot,
    },
    "2": {
        "name": "Few-shot",
        "category_prompt": CATEGORY_PROMPT_FEWSHOT,
        "subcategory_builder": build_subcategory_prompt_fewshot,
    },
    "3": {
        "name": "Chain-of-Thought",
        "category_prompt": CATEGORY_PROMPT_COT,
        "subcategory_builder": build_subcategory_prompt_cot,
    },
}

# ─────────────────────────────────────────────────────────────
# STEP 3 — LLM call helper with JSON enforcement + 1 retry
# ─────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str) -> dict:
    """
    Calls Groq LLM and parses a JSON response.
    Retries once if JSON parsing fails.
    """
    for attempt in range(2):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast, free, and actively supported on Groq
            temperature=0.1,       # low temperature → deterministic, structured output
            max_tokens=256,        # short responses for classification tasks
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        raw = response.choices[0].message.content.strip()

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            if attempt == 0:
                # Retry with an explicit reminder
                system_prompt += "\n\nREMINDER: Return ONLY valid JSON. Nothing else."
                continue
            raise ValueError(
                f"Model did not return valid JSON after 2 attempts.\nRaw: {raw}"
            )


# ─────────────────────────────────────────────────────────────
# STEP 4 — 2-call classification chain (Option B)
# ─────────────────────────────────────────────────────────────

def classify_document(text: str, category_prompt: str, subcategory_builder) -> dict:
    """
    Call 1: Determine category.
    Call 2: Determine sub-category constrained by the result of Call 1.
    Returns dict with category and sub_category.
    """
    # --- Call 1: Category ---
    cat_result = call_llm(
        system_prompt=category_prompt,
        user_message=f"Classify the following text:\n\n{text}",
    )
    category = cat_result.get("category", "").strip()

    # Validate category
    if category not in TAXONOMY:
        # Find closest match (simple inclusion check)
        for allowed in TAXONOMY:
            if allowed.lower() in category.lower() or category.lower() in allowed.lower():
                category = allowed
                break
        else:
            category = CATEGORIES[0]   # fallback

    # --- Call 2: Sub-category (constrained) ---
    subcategory_prompt = subcategory_builder(category)
    sub_result = call_llm(
        system_prompt=subcategory_prompt,
        user_message=f"Classify the sub-category for this text:\n\n{text}",
    )
    sub_category = sub_result.get("sub_category", "").strip()

    # Validate sub-category
    allowed_subs = TAXONOMY[category]
    if sub_category not in allowed_subs:
        for allowed in allowed_subs:
            if allowed.lower() in sub_category.lower():
                sub_category = allowed
                break
        else:
            sub_category = allowed_subs[0]  # fallback

    return {"category": category, "sub_category": sub_category}


# ─────────────────────────────────────────────────────────────
# STEP 5 — Load documents from dataset/ folder
# ─────────────────────────────────────────────────────────────

def load_documents(dataset_dir: str) -> list[dict]:
    """
    Reads all .txt files from the dataset directory.
    Returns a list of dicts: {document_name, text}
    """
    pattern = os.path.join(dataset_dir, "*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No .txt files found in: {dataset_dir}")

    documents = []
    for filepath in files:
        filename = Path(filepath).name
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append({"document_name": filename, "text": text})

    return documents


# ─────────────────────────────────────────────────────────────
# STEP 6 — Batch classification + save results
# ─────────────────────────────────────────────────────────────

def batch_classify(documents: list[dict], category_prompt: str, subcategory_builder) -> list[dict]:
    """
    Classifies each document and returns a list of result dicts.
    """
    results = []
    total = len(documents)

    for i, doc in enumerate(documents, start=1):
        name = doc["document_name"]
        text = doc["text"]

        print(f"  [{i:02d}/{total}] Classifying: {name} ...", end=" ", flush=True)

        try:
            classification = classify_document(text, category_prompt, subcategory_builder)
            result = {
                "document_name": name,
                "category":      classification["category"],
                "sub_category":  classification["sub_category"],
            }
            print(f"✅ {result['category']} → {result['sub_category']}")
        except Exception as e:
            result = {
                "document_name": name,
                "category":      "ERROR",
                "sub_category":  str(e),
            }
            print(f"❌ ERROR: {e}")

        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  GEN AI Upskilling — Exercise 02: Text Classifier")
    print("="*60)

    # Select prompting technique
    print("\nSelect a prompting technique:")
    for key, cfg in PROMPTS.items():
        print(f"  [{key}] {cfg['name']}")

    choice = input("\nYour choice (1/2/3): ").strip()
    if choice not in PROMPTS:
        print("Invalid choice. Defaulting to Zero-shot.")
        choice = "1"

    selected = PROMPTS[choice]
    print(f"\nTechnique selected: {selected['name']}")

    # Load dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    print(f"\nLoading documents from: {dataset_dir}")
    documents = load_documents(dataset_dir)
    print(f"Loaded {len(documents)} document(s).\n")

    # Classify all documents
    print("Starting classification (2-call chain: category → sub-category)...")
    print("-"*60)
    results = batch_classify(documents, selected["category_prompt"], selected["subcategory_builder"])

    # Save results.json
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "="*60)
    print(f"  ✅  Classification complete!")
    print(f"  📄  {len(results)} documents processed")
    print(f"  💾  Results saved to: results.json")
    print("="*60)
    print("\nFull results:\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
