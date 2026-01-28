import os
from recommender.recommend import (
    load_data,
    build_vector_store,
    generate_learning_path,
    analyze_skill_gap
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "resources_with_embeddings.json")

if __name__ == "__main__":
    user_profile = {
        "current_skills": ["Python", "Statistics"],
        "target_role": "Machine Learning Engineer",
        "experience_level": "Beginner"
    }

    resources = load_data(DATA_PATH)
    store = build_vector_store(resources)

    ranked_results = generate_learning_path(user_profile, store)
    skill_gap = analyze_skill_gap(
        user_profile["current_skills"], ranked_results
    )

    print("\n🎯 Personalized Learning Path (Explainable):\n")
    for i, item in enumerate(ranked_results, 1):
        res = item["resource"]
        exp = item["explanation"]
        print(
            f"{i}. {res['title']} ({res['level']}) | "
            f"Final Score: {item['final_score']:.3f}\n"
            f"   → Content similarity: {exp['content_similarity']}\n"
            f"   → Skill similarity: {exp['skill_similarity']}\n"
            f"   → Level match score: {exp['level_match']}\n"
        )

    print("🧠 Skill Gap Analysis:")
    for skill in skill_gap:
        print(f"- {skill}")
