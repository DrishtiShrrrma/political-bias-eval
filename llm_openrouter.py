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
                api_key="key"
            )
        else:
            raise ValueError(f"Provider '{provider}' not supported")

    def query(self, text: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        if self.provider == "cohere":
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.message.content[0].text

        elif self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider in ["google", "qwen"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Provider '{self.provider}' not supported")
