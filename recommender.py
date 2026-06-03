# ============================================================
#  Project 3 – AI Recommendation Logic
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Method  : Content-Based Filtering
#  Engine  : TF-IDF Vectorization + Cosine Similarity
#  Dataset : raw_skills.csv  (16 job roles, 8+ skills each)
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

# ── PHASE 1 : INPUT — Ingestion ──────────────────────────────
print("=" * 60)
print("  PROJECT 3 — Tech Stack Recommender (Content-Based AI)")
print("=" * 60)

# Load dataset
df = pd.read_csv("raw_skills.csv")
print(f"\n[DATASET]  {len(df)} job roles loaded from raw_skills.csv")

# ── PHASE 2 : PROCESS — TF-IDF Vectorization ────────────────

# Step 1 — Build the TF-IDF matrix from all job role skill sets
#           TF  : rewards terms frequent within one role
#           IDF : penalizes generic terms common across all roles
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["skills"])
print(f"[TF-IDF]   Vocabulary size : {len(vectorizer.vocabulary_)} unique skills")
print(f"           Matrix shape    : {tfidf_matrix.shape}  (roles × features)")

# ── USER INPUT — Minimum 3 skills (as per specification) ────
print("\n" + "─" * 60)
print("  TECH STACK RECOMMENDER — Enter your skills below")
print("  (minimum 3 skills for accurate matching)")
print("─" * 60)

raw_input_str = input("\nEnter your skills (comma-separated): ")

# Sanitize & parse input
user_skills_list = [s.strip().lower().replace(" ", "_")
                    for s in raw_input_str.split(",") if s.strip()]

if len(user_skills_list) < 3:
    print("\n⚠️  Warning: fewer than 3 skills entered. Accuracy may be reduced.")

user_profile_str = " ".join(user_skills_list)
print(f"\n[PROFILE]  Parsed skills : {user_skills_list}")

# ── STEP 2 : Scoring — Cosine Similarity ────────────────────
# Transform user profile into the same TF-IDF vector space
user_vector   = vectorizer.transform([user_profile_str])

# Calculate cosine similarity: measures ANGLE between vectors
# Range 0–1 → 1 = perfectly aligned, 0 = no overlap
scores        = cosine_similarity(user_vector, tfidf_matrix).flatten()
print(f"\n[SCORING]  Cosine similarity computed against all {len(df)} roles")

# ── STEP 3 : Sorting — Rank by score descending ─────────────
df["similarity_score"] = scores
df_sorted = df.sort_values("similarity_score", ascending=False)

# ── STEP 4 : Filtering — Top-N Output (prevent choice overload)
TOP_N = 3
top_results = df_sorted.head(TOP_N)

# ── PHASE 3 : OUTPUT — Display Top-N Recommendations ────────
print("\n" + "=" * 60)
print(f"  TOP {TOP_N} RECOMMENDED JOB ROLES FOR YOU")
print("=" * 60)

for rank, (_, row) in enumerate(top_results.iterrows(), start=1):
    match_pct = row["similarity_score"] * 100
    bar_len   = int(match_pct / 5)          # scale to 20-char bar
    bar       = "█" * bar_len + "░" * (20 - bar_len)
    print(f"\n  #{rank}  {row['job_role']}")
    print(f"       Match  : {bar}  {match_pct:.1f}%")
    print(f"       Skills : {row['skills']}")

print("\n" + "─" * 60)
print("  ✅ Recommendation engine complete.")
print(f"  Your profile : {user_skills_list}")
print("─" * 60)
