# Personalized Learning Path Generator using ENDEE 🚀

An explainable, multi-vector recommendation system that generates personalized learning paths based on a user’s current skills, target role, and experience level — powered by **ENDEE as a vector database**.

---

## 🔍 Why This Project?

Most learning platforms recommend content using:
- keyword matching
- static roadmaps
- one-size-fits-all paths

This project solves that by using **semantic understanding**, **skill-aware reasoning**, and **explainable ranking**, making recommendations both **personalized** and **transparent**.

---

## 🧠 Key Features

- ✅ Semantic recommendations using vector embeddings  
- ✅ **Multiple vector indexes** (content + skill representations)  
- ✅ Hybrid ranking (ML + rule-based logic)  
- ✅ Explainable scoring for every recommendation  
- ✅ Explicit **skill gap analysis**  
- ✅ Cold-start friendly design  
- ✅ Clean, modular architecture  

---

## 🏗️ System Architecture

User Profile
↓
Embedding Model
↓
ENDEE Vector Database
├── Content Index (what the resource is about)
└── Skill Index (what the resource teaches)
↓
Hybrid Ranker
↓
Explainable Learning Path + Skill Gap Analysis


---

## ⚙️ How ENDEE Is Used

This project uses ENDEE as a **vector database** to store and retrieve embeddings efficiently.

### 🔹 Multiple Vector Indexes (Advanced Usage)
Each learning resource is represented using **two separate embeddings**:

- **Content embedding** → captures semantic meaning of the resource  
- **Skill embedding** → captures the skills taught by the resource  

Both indexes are queried independently and merged using a weighted scoring strategy.

This mirrors how vector databases are used in **production-grade AI systems**.

---

## 🧮 Hybrid Ranking Strategy

Final ranking score is computed as:

Final Score =
0.6 × Content Similarity
0.2 × Skill Similarity
0.2 × Level Compatibility


This ensures:
- semantic relevance
- skill alignment
- proper difficulty progression

Low-quality matches are filtered using a similarity threshold.

---

## 🧠 Explainability (Interview-Ready)

Every recommendation includes a transparent breakdown:
- content similarity score
- skill similarity score
- level match score
- final aggregated score

This makes the system:
- debuggable
- trustworthy
- easy to reason about

---

## 🛠️ Tech Stack

- **Python**
- **Sentence Transformers**
- **NumPy**
- **ENDEE (Vector Database)**

---

## Sample Output 

<img width="1007" height="577" alt="Screenshot 2026-01-28 200725" src="https://github.com/user-attachments/assets/5693f9aa-fe49-4af9-8b60-76f490fb04a3" />

<img width="1013" height="982" alt="Screenshot 2026-01-28 200713" src="https://github.com/user-attachments/assets/c1a645f3-aecc-42a6-a5c7-44c19c3f1f2f" />


## ▶️ How to Run

```bash
# Create embeddings
python embeddings/generate_embeddings.py

# Run the recommender
python app.py
