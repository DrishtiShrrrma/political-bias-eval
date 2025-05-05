import os
import cohere
import mistralai
from openai import OpenAI

class LLM:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

        if provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            self.client = cohere.ClientV2(api_key)
        elif provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            self.client = mistralai.Mistral(api_key)
        elif provider in ["google", "qwen"]:
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="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"
            )
        else:
            raise ValueError(f"Provider '{provider}' not supported")

    def query(self, text: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        instruction = (
            "Write a formal article starting directly with the content. "
            "Do not include phrases like 'Here is your report' or any commentary about the request either at the beginning or the end. "
        )

        prompt = f"{instruction}\n\n{text}"

        if self.provider == "cohere":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.message.content[0].text

        elif self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider in ["google", "qwen"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Provider '{self.provider}' not supported")