import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Load Gemini 2.0 Flash model
model = genai.GenerativeModel("gemini-2.0-flash")

# Function to identify high-level prerequisites using Gemini
def identify_prerequisites_for_question(question):
    prompt = f"""
You are an expert math educator. Given the following math word problem, identify the key mathematical concepts or reasoning skills that are essential to solve it.

Problem:
{question}

Instructions:
- List 3 to 5 high-level but specific mathematical concepts required to solve the problem
- Avoid vague terms like "Arithmetic" — instead, use terms such as Addition, Subtraction, Multiplication, Division, Fractions, Estimation, Ratios, Word Problem Comprehension, etc.
- Avoid overly detailed techniques or procedural steps
- Only include concepts that are genuinely essential
- Return the concepts as a comma-separated list (e.g., "Addition, Subtraction, Multiplication")
- Do not include numbering or explanations — just the concepts
- If no math concepts are required, return "None"

Essential Prerequisites:"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        if raw.lower() == "none":
            return ["None"]

        prerequisites = [
            concept.strip()
            for concept in raw.split(",")
            if concept.strip()
        ]

        return prerequisites
    except Exception as e:
        print("Error for question:", question)
        print("Exception:", e)
        return ["Error"]
