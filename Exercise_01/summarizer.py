"""
GEN AI Upskilling - Exercise 1
Text Summarization with Structured JSON Output using Groq
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Load environment variables from .env file
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# STEP 1 — Instantiate the LLM (Groq client)
# ─────────────────────────────────────────────────────────────
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Prompt design (3 techniques available)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_ZEROSHOT = """
You are an expert text summarization assistant.
Your task is to analyze the user's text and return a summary.

IMPORTANT: Respond ONLY with a valid JSON object. No extra text, no markdown, no explanations.

Required JSON format:
{
  "summary": "brief summary of the input text in 2-3 sentences",
  "key_points": ["key point 1", "key point 2", "key point 3"],
  "sentiment": "positive | negative | neutral",
  "word_count": number_of_words_in_original_text,
  "technique": "zero-shot"
}
"""

SYSTEM_PROMPT_FEWSHOT = """
You are an expert text summarization assistant.
Respond ONLY with a valid JSON object. No extra text, no markdown.

Here are two examples of the expected input and output:

Example 1:
Input: "Python is a popular programming language known for its clear and readable syntax. It is widely used in data science, web development, and automation."
Output: {"summary": "Python is a versatile language praised for readability, used in many fields.", "key_points": ["popular language", "clear syntax", "used in data science and web"], "sentiment": "positive", "word_count": 26, "technique": "few-shot"}

Example 2:
Input: "Air pollution in major cities is causing serious health problems. Governments are struggling to implement effective regulations."
Output: {"summary": "Urban air pollution poses health risks while regulatory responses remain insufficient.", "key_points": ["air pollution in cities", "health problems", "weak government regulation"], "sentiment": "negative", "word_count": 22, "technique": "few-shot"}

Now apply the same pattern to the user's text. Return ONLY the JSON object.
"""

SYSTEM_PROMPT_COT = """
You are an expert text summarization assistant.

Follow these internal reasoning steps before producing the output:
  Step 1: Identify the main topic of the text.
  Step 2: Extract the 2-3 most important points.
  Step 3: Determine the overall sentiment (positive, negative, or neutral).
  Step 4: Write a concise summary in 2-3 sentences.
  Step 5: Count the approximate number of words in the original text.

After completing all reasoning steps, respond ONLY with the following valid JSON object.
Do NOT include the reasoning steps in your response.

Required JSON format:
{
  "summary": "concise summary in 2-3 sentences",
  "key_points": ["key point 1", "key point 2", "key point 3"],
  "sentiment": "positive | negative | neutral",
  "word_count": approximate_word_count,
  "technique": "chain-of-thought"
}
"""

PROMPTS = {
    "1": ("Zero-shot",     SYSTEM_PROMPT_ZEROSHOT),
    "2": ("Few-shot",      SYSTEM_PROMPT_FEWSHOT),
    "3": ("Chain of thought", SYSTEM_PROMPT_COT),
}


# ─────────────────────────────────────────────────────────────
# STEP 3 — Core function: call Groq and enforce JSON output
# ─────────────────────────────────────────────────────────────

def summarize_text(user_text: str, system_prompt: str) -> dict:
    """
    Sends the user text to the Groq LLM and returns a parsed JSON dict.
    Raises ValueError if the model response is not valid JSON.
    """
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",         # fast and free on Groq
        temperature=0.3,                 # lower = more deterministic output
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{user_text}"
            }
        ]
    )

    raw_response = chat_completion.choices[0].message.content.strip()

    # STEP 3 — Parse and validate the JSON response
    try:
        # Clean up markdown code fences if the model adds them
        clean = raw_response.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON.\n"
            f"Raw response:\n{raw_response}\n"
            f"JSON error: {e}"
        )


# ─────────────────────────────────────────────────────────────
# STEP 4 — Accept user input and run the app
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  GEN AI Upskilling — Exercise 1: Text Summarizer")
    print("="*55)

    # Choose prompting technique
    print("\nSelect a prompting technique:")
    for key, (name, _) in PROMPTS.items():
        print(f"  [{key}] {name}")

    choice = input("\nYour choice (1/2/3): ").strip()
    if choice not in PROMPTS:
        print("Invalid choice. Defaulting to Zero-shot.")
        choice = "1"

    technique_name, system_prompt = PROMPTS[choice]
    print(f"\nTechnique selected: {technique_name}")

    # Accept text input
    print("\nPaste the text you want to summarize.")
    print("(Press Enter twice when done)\n")

    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    user_text = "\n".join(lines).strip()

    if not user_text:
        print("No text provided. Exiting.")
        return

    print("\nSending request to Groq...")

    try:
        result = summarize_text(user_text, system_prompt)

        print("\n" + "="*55)
        print("  ✅  Valid JSON response received")
        print("="*55)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("="*55 + "\n")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
