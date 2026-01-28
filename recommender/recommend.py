import json
from sentence_transformers import SentenceTransformer
from endee.vector_store import EndeeVectorStore

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

LEVEL_ORDER = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
SIMILARITY_THRESHOLD = 0.3  # Interview flex 🔥

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def build_vector_store(resources):
    store = EndeeVectorStore()
    for r in resources:
        store.add(r["content_embedding"], r["skill_embedding"], r)
    return store

def level_score(user_level, resource_level):
    return 1.0 if LEVEL_ORDER[user_level] >= LEVEL_ORDER[resource_level] else 0.5

def generate_learning_path(user_profile, store):
    intent_text = (
        f"User wants to become a {user_profile['target_role']} "
        f"at {user_profile['experience_level']} level"
    )
    skill_text = " ".join(user_profile["current_skills"])

    intent_emb = model.encode(intent_text)
    skill_emb = model.encode(skill_text)

    content_results = store.query_content(intent_emb)
    skill_results = store.query_skill(skill_emb)

    merged = {}

    for score, res in content_results:
        if score >= SIMILARITY_THRESHOLD:
            merged[res["id"]] = {
                "resource": res,
                "content_score": score,
                "skill_score": 0.0
            }

    for score, res in skill_results:
        if res["id"] in merged:
            merged[res["id"]]["skill_score"] = score

    ranked = []
    for item in merged.values():
        res = item["resource"]
        final_score = (
            0.6 * item["content_score"] +
            0.2 * item["skill_score"] +
            0.2 * level_score(
                user_profile["experience_level"], res["level"]
            )
        )

        ranked.append({
            "resource": res,
            "final_score": final_score,
            "explanation": {
                "content_similarity": round(item["content_score"], 3),
                "skill_similarity": round(item["skill_score"], 3),
                "level_match": level_score(
                    user_profile["experience_level"], res["level"]
                )
            }
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked

def analyze_skill_gap(user_skills, ranked_results):
    covered = set()
    for r in ranked_results:
        covered.update(r["resource"]["skills_covered"])
    return sorted(list(covered - set(user_skills)))
