import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('models/gemini-1.5-flash')

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text
