import os
import google.generativeai as genai

# ✅ Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_legal_issue(user_input, similar_cases):
    # Convert similar cases into a readable format
    case_summaries = "\n".join([f"- {c['title']} ({c['jurisdiction']}): {c['summary']}" for c in similar_cases])

    prompt = f"""
    You are a legal research assistant.
    User problem: {user_input}

    Similar cases retrieved:
    {case_summaries}

    Tasks:
    1. Identify the main legal issues.
    2. Suggest relevant laws, statutes, or references.
    3. Provide structured output.

    Respond strictly in JSON format:
    {{
      "issues": ["..."],
      "references": ["..."]
    }}
    """

    # ✅ Call Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")  # or gemini-1.5-pro for more advanced
    response = model.generate_content(prompt)

    # Gemini sometimes wraps JSON in text → clean it
    content = response.text.strip()
    if content.startswith("```"):
        content = content.split("```")[1].replace("json", "").strip()

    return content
