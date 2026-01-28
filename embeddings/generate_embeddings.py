import json
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "learning_resources.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "resources_with_embeddings.json")

def load_resources(path):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    resources = load_resources(DATA_PATH)

    content_texts = [
        f"{r['title']} - {r['description']} ({r['level']})"
        for r in resources
    ]

    skill_texts = [
        " ".join(r["skills_covered"])
        for r in resources
    ]

    content_embeddings = model.encode(content_texts, convert_to_numpy=True)
    skill_embeddings = model.encode(skill_texts, convert_to_numpy=True)

    for r, c_emb, s_emb in zip(resources, content_embeddings, skill_embeddings):
        r["content_embedding"] = c_emb.tolist()
        r["skill_embedding"] = s_emb.tolist()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(resources, f, indent=2)

    print("✅ Content & Skill embeddings generated and saved.")
