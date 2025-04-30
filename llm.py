import os
import torch
import cohere
import mistralai
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_float32_matmul_precision("high")
torch._dynamo.disable()


class LLM:
    def __init__(self, provider, model):
        self.model = model
        self.provider = provider

        if provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            self.client = cohere.ClientV2(api_key)

        elif provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            self.client = mistralai.Mistral(api_key)

        elif provider in ["google", "qwen"]:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.client = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            ).to(self.device)

        else:
            raise ValueError(f"Provider '{provider}' is not supported.")

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
            messages = [{"role": "user", "content": text}]
            chat_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(chat_input, return_tensors="pt").to(self.client.device)

            generated_ids = self.client.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Gemma-specific cleanup
            if "gemma" in self.model.lower():
                if "<start_of_turn>model" in output_text:
                    return output_text.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()
                else:
                    return output_text.strip()
            else:
                return output_text.strip()

        else:
            raise ValueError(f"Provider '{self.provider}' is not supported.")
