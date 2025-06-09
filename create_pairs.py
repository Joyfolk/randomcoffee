import argparse
import asyncio
import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import dotenv
import numpy as np
import networkx as nx
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

# Default models
DEFAULT_CHAT_MODEL = "gpt-4"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

@dataclass
class UserProfile:
    username: str
    intro_text: str
    link: str
    embedding: np.ndarray
    pairing_history: List[str] = None

    def __post_init__(self):
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)

        if self.pairing_history is None:
            self.pairing_history = []

class CoffeePairingSystem:
    def __init__(self, embeddings_file: str, 
                 randomness_factor: float = 0.3,
                 pairs_dir: str = None):
        self.embeddings_file = embeddings_file
        self.randomness_factor = randomness_factor
        self.pairs_dir = pairs_dir
        self.profiles = []

        os.makedirs(self.pairs_dir, exist_ok=True)

    def load_profiles(self) -> List[UserProfile]:
        """Load user profiles from embeddings file"""
        try:
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.profiles = []
            for username, user_data in data.items():
                if not user_data.get("embedding"):
                    print(f"Skipping {username} - no valid embedding")
                    continue

                profile = UserProfile(
                    username=username,
                    intro_text=user_data["intro_text"],
                    link=user_data["link"],
                    embedding=user_data["embedding"],
                    pairing_history=[],
                )
                self.profiles.append(profile)

            self._load_pairing_history()

            print(f"Loaded {len(self.profiles)} user profiles")
            return self.profiles

        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return []

    def _load_pairing_history(self):
        """Load pairing history from CSV files in the pairs directory"""
        if not os.path.exists(self.pairs_dir):
            return

        previous_pairings = sorted([f for f in os.listdir(self.pairs_dir) if f.endswith('.csv') and f.startswith('pairs_')])

        username_to_profile = {profile.username: profile for profile in self.profiles}

        for file_name in previous_pairings:
            file_path = os.path.join(self.pairs_dir, file_name)
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    # Skip header
                    next(reader, None)

                    for row in reader:
                        if len(row) < 4:
                            print(f"Skipping invalid row in {file_name}: {row}")
                            continue

                        user1, _, user2, _ = row[:4]

                        if user1 in username_to_profile:
                            profile1 = username_to_profile[user1]
                            profile1.pairing_history.append(user2)
                        if user2 in username_to_profile:
                            profile2 = username_to_profile[user2]
                            profile2.pairing_history.append(user1)

            except Exception as e:
                print(f"Error loading pairing history from {file_path}: {e}")

    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate similarity matrix between all users"""
        embeddings = np.array([profile.embedding for profile in self.profiles])
        return cosine_similarity(embeddings)

    def create_optimal_pairing(self) -> List[Tuple[UserProfile, UserProfile, float]]:
        """Find optimal matching of all possible pairs, regardless of history or threshold.
        If the number of users is odd, pair the last user with the best available partner (even if duplicated).
        """
        similarity_matrix = self.calculate_similarity_matrix()
        n_users = len(self.profiles)
        graph = nx.Graph()

        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity_score = similarity_matrix[i][j]
                random_factor = random.random() * self.randomness_factor
                final_score = similarity_score + random_factor
                graph.add_edge(i, j, weight=final_score)

        # Find the maximum weight matching
        matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True)
        pairs = []
        paired_indices = set()
        for i, j in matching:
            user1, user2 = self.profiles[i], self.profiles[j]
            score = graph[i][j]['weight']
            pairs.append((user1, user2, score))
            paired_indices.add(i)
            paired_indices.add(j)

        # If odd number of users, pair the unpaired user with the best available partner (even if duplicated)
        if len(paired_indices) < n_users:
            unpaired_idx = next(idx for idx in range(n_users) if idx not in paired_indices)
            best_score = -1
            best_partner_idx = None
            for idx in range(n_users):
                if idx == unpaired_idx:
                    continue
                score = similarity_matrix[unpaired_idx][idx]
                if score > best_score:
                    best_score = score
                    best_partner_idx = idx
            if best_partner_idx is not None:
                user1 = self.profiles[unpaired_idx]
                user2 = self.profiles[best_partner_idx]
                random_factor = random.random() * self.randomness_factor
                final_score = best_score + random_factor
                pairs.append((user1, user2, final_score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def save_pairing(self, csv_data: List[List[str]]) -> str | None:
        """Save pairing history to a new CSV file in the pairs directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pairings_file = os.path.join(self.pairs_dir, f"pairs_{timestamp}.csv")

        try:
            with open(pairings_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['telegram_user_1', 'telegram_user_1_url', 'telegram_user_2', 'telegram_user_2_url', 'pairing_explanation_message'])
                for row in csv_data:
                    writer.writerow(row)
            return pairings_file
        except Exception as e:
            print(f"Error saving pairing history: {e}")
            return None

class PairingMessageGenerator:
    def __init__(self, model: str = "gpt-4", templates_dir: str = "templates"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model
        self.templates_dir = templates_dir
        self.templates = self._load_templates()

    def _load_templates(self):
        """Load templates from the templates directory"""
        templates = {
            "system": "",
            "pairing": ""
        }

        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' not found")

        system_path = os.path.join(self.templates_dir, "system_prompt.txt")
        if os.path.exists(system_path):
            with open(system_path, 'r', encoding='utf-8') as f:
                templates["system"] = f.read().strip()
        else:
            raise FileNotFoundError(f"System prompt template '{system_path}' not found")

        pairing_path = os.path.join(self.templates_dir, "pairing_message.txt")
        if os.path.exists(pairing_path):
            with open(pairing_path, 'r', encoding='utf-8') as f:
                templates["pairing"] = f.read().strip()
        else:
            raise FileNotFoundError(f"Pairing message template '{pairing_path}' not found")

        print(f"Loaded templates: system, pairing")
        return templates

    async def generate_pairing_message(self, user1: UserProfile, user2: UserProfile, 
                                     similarity_score: float) -> str:
        """Generate a personalized pairing message using templates"""
        prompt = self.templates["pairing"]
        prompt = prompt.replace("{user1}", user1.username)
        prompt = prompt.replace("{user2}", user2.username)
        prompt = prompt.replace("{intro1}", user1.intro_text[:300])
        prompt = prompt.replace("{intro2}", user2.intro_text[:300])
        prompt = prompt.replace("{score:.2f}", f"{similarity_score:.2f}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.templates["system"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Error generating pairing message: {e}")

async def main():
    parser = argparse.ArgumentParser(description='Create coffee pairs from user embeddings')
    parser.add_argument('--embeddings', '-e', default='data/embeddings.json',
                        help='Embeddings file (default: data/embeddings.json)')
    parser.add_argument('--randomness', '-r', type=float, default=0.2,
                        help='Randomness factor (default: 0.2)')
    parser.add_argument('--model', '-m', default=None,
                        help='LLM model for generating pairing messages (default: from CHAT_MODEL env var or gpt-4)')
    parser.add_argument('--pairs-dir', default='data/pairs',
                        help='Directory for storing pairing CSV files (default: data/pairs)')
    parser.add_argument('--templates-dir', default='templates',
                        help='Directory containing message templates (default: templates)')
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is required for generating messages")

    pairing_system = CoffeePairingSystem(
        embeddings_file=args.embeddings,
        randomness_factor=args.randomness,
        pairs_dir=args.pairs_dir
    )

    profiles = pairing_system.load_profiles()

    if len(profiles) < 2:
        print("Need at least 2 user profiles to create pairs!")
        return

    pairs = pairing_system.create_optimal_pairing()

    print(f"\nCreated {len(pairs)} coffee pairs:")
    print("=" * 50)

    model = args.model or os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    message_generator = PairingMessageGenerator(model=model, templates_dir=args.templates_dir)

    csv_data = []
    for i, (user1, user2, score) in enumerate(pairs, 1):
        print(f"\nPair #{i}: {user1.username} & {user2.username}")
        print(f"Similarity Score: {score:.3f}")
        print(f"Links: {user1.link} | {user2.link}")
        message = await message_generator.generate_pairing_message(user1, user2, score)
        print(f"Pairing Message:\n{message}")

        csv_row = [
            user1.username,
            user1.link,
            user2.username,
            user2.link,
            message
        ]
        csv_data.append(csv_row)

    pairing_file = pairing_system.save_pairing(csv_data)
    print(f"Pairings saved to {pairing_file}")

if __name__ == "__main__":
    asyncio.run(main())
