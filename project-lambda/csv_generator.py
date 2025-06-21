#Developed By : Sriram P

import pandas as pd
import time
from datasets import load_dataset
from tqdm import tqdm
from gemini_utils import identify_prerequisites_for_question

# Load GSM8K dataset (first 200 questions)
gsm8k = load_dataset("gsm8k", "main", split="train[:200]")

# Process and collect results
results = []

print("Processing GSM8K questions with Gemini...")
for entry in tqdm(gsm8k):
    question = entry["question"]
    prerequisites = identify_prerequisites_for_question(question)
    results.append({
        "question": question,
        "prerequisites": prerequisites
    })
    time.sleep(5)

# Convert list to comma-separated string for saving
df = pd.DataFrame(results)
df["prerequisites"] = df["prerequisites"].apply(lambda x: ", ".join(x))
df.to_csv("gsm8k_prerequisites_gemini_final.csv", index=False)

print("Saved results to gsm8k_prerequisites_gemini_final.csv")
