from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI

try:
    from llm.config import API_KEY, MODEL_NAME
except ImportError:
    from .config import API_KEY, MODEL_NAME


class GeminiLLM:
    def __init__(self):
        self.llm = GoogleGenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
        )
        Settings.llm = self.llm

    def get_llm(self):
        return self.llm


Gemini = GeminiLLM
