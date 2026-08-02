try:
    from llm.gemini import GeminiLLM
    from llm.prompts import PROMPT, build_prompt
except ImportError:
    from graphRAG.llm.gemini import GeminiLLM
    from graphRAG.llm.prompts import PROMPT, build_prompt


class AnswerGenerator:

    def __init__(self, llm=None):
        self.llm = llm or GeminiLLM().get_llm()

    def generate(self, question: str, contexts) -> str:
        context = "\n\n".join(
            getattr(item, "text", str(item)) for item in contexts
        )

        prompt = build_prompt(
            context=context,
            question=question,
        )

        full_prompt = (
            PROMPT
            + "\n\n"
            + prompt
        )

        return self.llm.complete(full_prompt).text
