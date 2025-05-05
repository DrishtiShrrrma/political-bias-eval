import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParagraphGenerator:
    def __init__(self, prompts_file, provider, model, output_dir="results"):
        self.prompts = self.load_prompts(prompts_file)
        self.output_dir = output_dir
        self.llm = LLM(provider, model)

    def load_prompts(self, prompts_file):
        with open(prompts_file, encoding="utf-8") as f:
            data = json.load(f)
        if "metadata" not in data or "prompts" not in data:
            raise ValueError("Invalid prompts file format")
        return data["prompts"]

    def generate_all(self, provider, model, max_tokens=1000):
        """Multithreaded version of generate_all that tracks prompt-level progress"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Count all prompts for the progress bar
        total = sum(
            len(prompts)
            for topic in self.prompts.values()
            for lang in topic.values()
            for prompts in lang.values()
        )

        tasks = []
        with tqdm(total=total, desc="Generating paragraphs") as pbar:
            with ThreadPoolExecutor(max_workers=20) as executor:
                for topic_name, topic_data in self.prompts.items():
                    for lang, lang_data in topic_data.items():
                        for stance, prompts in lang_data.items():
                            tasks.append(executor.submit(
                                self._generate_stance_threadsafe,
                                provider, model, topic_name, lang, stance, prompts, max_tokens, pbar
                            ))

                for future in as_completed(tasks):
                    try:
                        future.result()
                    except Exception as e:
                        print("Error during generation:", e)

    def _generate_stance_threadsafe(self, provider, model, topic, language, stance, prompts, max_tokens, pbar):
        """Wraps _generate_stance for use in threaded execution"""
        self._generate_stance(
            provider=provider,
            model=model,
            topic=topic,
            language=language,
            stance=stance,
            prompts=prompts,
            max_tokens=max_tokens,
            pbar=pbar
        )


    def _generate_stance(self, provider, model, topic, language, stance, prompts, max_tokens, pbar):
        model_clean = model.split("/")[-1]

        for i, prompt in enumerate(prompts):
            path = os.path.join(
                self.output_dir,
                language,
                topic,
                provider,
                model_clean,
                stance,
                f"sample_{i+1}.txt"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Call the LLM
            response = self.llm.query(
                text=prompt,
                max_tokens=max_tokens
            )

            # Save output
            with open(path, "w", encoding="utf-8") as f:
                f.write(response)

            # Update progress bar
            pbar.update(1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate paragraphs from prompts")
    parser.add_argument("--prompts", required=True, help="Input prompts JSON file")
    parser.add_argument("--provider", default="cohere", help="LLM provider")
    parser.add_argument("--model", default="command-a-03-2025", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max output tokens")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    generator = ParagraphGenerator(args.prompts, args.provider, args.model, args.output)
    generator.generate_all(
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens
    )
