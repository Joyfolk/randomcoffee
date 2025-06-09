import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict

import dotenv
from openai import OpenAI

# API delay to avoid rate limiting
API_DELAY = 0.1
# Default model and provider settings
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

dotenv.load_dotenv()

class IntroEmbeddingGenerator:
    def __init__(self, intros_dir: str, output_file: str, model: str, base_url: str, api_key: str):
        self.intros_dir = Path(intros_dir)
        self.output_file = output_file
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        print(f"Using API at: {base_url}")
        print(f"Using model: {model}")

    def load_intros(self) -> Dict[str, Dict]:
        """Load all intro files and extract text content"""
        intros = {}

        if not self.intros_dir.exists():
            raise FileNotFoundError(f"Directory {self.intros_dir} not found!")

        for file_path in self.intros_dir.glob("*.txt"):
            username = file_path.stem

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if len(lines) >= 2:
                    link = lines[0].strip()
                    intro_text = ''.join(lines[1:]).strip()

                    intros[username] = {
                        "username": username,
                        "link": link,
                        "intro_text": intro_text
                    }

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(intros)} user intros")
        return intros

    async def generate_embeddings(self, intros: Dict[str, Dict]) -> Dict[str, Dict]:
        """Generate embeddings for all intros"""
        total = len(intros)
        processed = 0

        for username, data in intros.items():
            try:
                response = self.client.embeddings.create(
                    input=data["intro_text"],
                    model=self.model
                )
                embedding = response.data[0].embedding

                data["embedding"] = embedding

                processed += 1
                if processed % 10 == 0 or processed == total:
                    print(f"Progress: {processed}/{total} embeddings generated")

                time.sleep(API_DELAY)

            except Exception as e:
                print(f"Error generating embedding for {username}: {e}")
                data["embedding"] = []

        return intros

    def save_embeddings(self, data: Dict[str, Dict]) -> None:
        """Save embeddings to JSON file"""
        serializable_data = {}

        for username, user_data in data.items():
            serializable_data[username] = {
                "username": user_data["username"],
                "link": user_data["link"],
                "intro_text": user_data["intro_text"],
                "embedding": user_data["embedding"],
                "model_used": self.model,
                "pairing_history": user_data.get("pairing_history", []),
                "last_paired": user_data.get("last_paired", None)
            }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Embeddings saved to {self.output_file}")

async def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for user intros')
    parser.add_argument('--input', '-i', default='data/intros',
                        help='Directory containing intro files (default: data/intros)')
    parser.add_argument('--output', '-o', default='data/embeddings.json',
                        help='Output file for embeddings (default: data/embeddings.json)')
    parser.add_argument('--model', '-m', default=None,
                        help='Embedding model to use (default: from EMBEDDING_MODEL env var or text-embedding-3-small)')
    parser.add_argument('--base-url', '-b', default=None,
                        help='API base URL (default: from OPENAI_API_BASE_URL env var or OpenAI API)')
    args = parser.parse_args()

    base_url = args.base_url or os.getenv("OPENAI_API_BASE_URL", DEFAULT_BASE_URL)
    model = args.model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required")

    generator = IntroEmbeddingGenerator(
        intros_dir=args.input,
        output_file=args.output,
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    intros = generator.load_intros()
    embedded_data = await generator.generate_embeddings(intros)
    generator.save_embeddings(embedded_data)

    print("Embedding generation complete!")

if __name__ == "__main__":
    asyncio.run(main())
