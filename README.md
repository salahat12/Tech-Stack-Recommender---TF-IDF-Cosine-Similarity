# 🎯 Tech Stack Recommender — AI Recommendation Logic

> **Project 3** of the DecodeLabs Industrial Training Program | Batch 2026  
> *From passive classification to active prediction — the Digital Matchmaker.*

---

## 📌 Overview

A content-based recommendation engine that maps a user's raw skills to the most relevant tech career paths. Built using **TF-IDF Vectorization** and **Cosine Similarity**, it transforms qualitative skill tags into mathematical vectors and finds the highest-aligned job roles — no user history required.

---

## 🧠 The Concept: Curing Choice Overload

Recommendation engines like Netflix and Amazon solve the same problem: too many options, too little signal. This project builds the logic skeleton behind those systems.

| | Project 2 (Classification) | Project 3 (Recommendation) |
|---|---|---|
| **Goal** | Label what data IS | Predict what user WANTS |
| **Input** | Fixed feature vector | Free-form user profile |
| **Output** | Single class | Ranked Top-N list |
| **Engine** | KNN distance | Cosine similarity |

---

## ⚙️ Architecture: The 4-Step Ranking Pipeline

```
INPUT               PROCESS                         OUTPUT
──────────────────────────────────────────────────────────
User Skills    →  1. Ingestion  (parse 3+ skills)  →  Top-3
raw_skills.csv    2. Scoring    (TF-IDF + Cosine)     Ranked
                  3. Sorting    (descending score)    Job
                  4. Filtering  (Top-N truncation)    Roles
```

---

## 🔬 How TF-IDF Works

Binary matching (0/1) treats all tags equally — **TF-IDF fixes this**:

- **TF (Term Frequency):** Terms appearing often within one role get higher weight → they describe that role precisely.
- **IDF (Inverse Document Frequency):** Terms appearing across many roles get penalized → generic terms like "python" matter less than "tensors".

```
TF  = count(term in role) / total terms in role
IDF = log(total roles / roles containing term)
TF-IDF = TF × IDF
```

---

## 📐 Why Cosine Similarity, Not Euclidean Distance?

Euclidean distance is sensitive to **magnitude** — a role with 10 skills will always seem "farther" than one with 3 skills, even if they're identical in direction. Cosine similarity measures the **angle** between two vectors, ignoring size entirely.

```
cos(θ) = (A · B) / (||A|| × ||B||)

Score 1.0 → Perfect alignment
Score 0.0 → No shared features
Score -1  → Opposite directions
```

---

## ✅ Features

- 📥 **Ingestion** — Accepts 3+ comma-separated skills; sanitizes and normalizes input
- 🧮 **TF-IDF Vectorization** — Builds a 67-feature weighted vocabulary space from 16 job roles
- 📐 **Cosine Similarity Scoring** — Magnitude-invariant similarity for accurate matching
- 📊 **Visual Match Bar** — Progress bar showing match % for each recommendation
- 🔝 **Top-3 Filtering** — Prevents choice overload; shows only highest-scoring matches
- ❄️ **Cold Start Handling** — Onboarding via explicit skill input bootstraps the user vector

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install scikit-learn pandas numpy
```

### Run the recommender

```bash
python3 recommender.py
```

### Example Session

```
============================================================
  PROJECT 3 — Tech Stack Recommender (Content-Based AI)
============================================================

[DATASET]  16 job roles loaded from raw_skills.csv
[TF-IDF]   Vocabulary size : 67 unique skills
           Matrix shape    : (16, 67)  (roles × features)

Enter your skills (comma-separated): python, machine_learning, sql

[PROFILE]  Parsed skills : ['python', 'machine_learning', 'sql']
[SCORING]  Cosine similarity computed against all 16 roles

============================================================
  TOP 3 RECOMMENDED JOB ROLES FOR YOU
============================================================

  #1  Data Scientist
       Match  : █████████░░░░░░░░░░░  48.3%
       Skills : python sql machine_learning data_analysis ...

  #2  ML Engineer
       Match  : ███████░░░░░░░░░░░░░  37.5%
       Skills : python tensorflow pytorch deep_learning ...

  #3  NLP Engineer
       Match  : ██████░░░░░░░░░░░░░░  34.9%
       Skills : python nlp machine_learning transformers ...
```

---

## 📁 Project Structure

```
decodelabs-ai-projects/
│
├── Project_1/
│   ├── chatbot.py
│   └── README.md
│
├── Project_2/
│   ├── classifier.py
│   └── README.md
│
├── Project_3/
│   ├── recommender.py     ← Main recommendation engine
│   ├── raw_skills.csv     ← Dataset: 16 roles × 8+ skills
│   └── README.md          ← You are here
```

---

## 🔑 Key Concepts Demonstrated

- **Content-Based Filtering** — Match user profile to item attributes, no other-user data needed
- **TF-IDF Vectorization** — Weighted feature extraction that rewards specificity
- **Cosine Similarity** — Angle-based matching invariant to vector magnitude
- **Vector Space Model** — User and items live in the same mathematical vocabulary space
- **Top-N Filtering** — Truncate output to prevent choice overload
- **Cold Start Solution** — Explicit onboarding survey (3+ skills) bootstraps the user vector

---

## 🛣️ What's Next (Project 4 Preview)

Project 3 uses **content** (skills tags) to recommend.  
The next level adds **behavior** — collaborative filtering learns from what users *do*, not just what they *say*.

```
Project 3:  Skills tags  →  TF-IDF + Cosine  →  Career match  (Content)
Project 4:  User ratings →  Matrix factoring  →  "Users like you liked..."  (Collaborative)
```

---

## 🏷️ Built With

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Dataset-green?style=flat-square&logo=pandas)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Batch_2026-purple?style=flat-square)

---

## 📄 License

This project is part of the **DecodeLabs Industrial Training Kit**.  
Built for educational and portfolio purposes.

---

*"We do not write the rules. We provide history, and the machine derives the logic."*  
**— DecodeLabs, Module 03**
