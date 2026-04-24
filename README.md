# 🎵 Melodia — Hybrid Music Recommendation System

A three-signal hybrid recommender that combines **Collaborative Filtering**, **Content-Based Audio Analysis**, and **MMR Diversity Re-ranking** — deployed as an interactive Streamlit web app.

---

## What it does

- Recommends tracks personalised to each user's listening history
- Balances **familiarity** (songs similar to what you've heard) with **discovery** (genuinely new tracks)
- Explains every recommendation in plain English: *"Similar energy to 'Creep' by Radiohead, which you played often — new artist for you."*
- Adapts automatically to cold-start users (few listens) and heavy listeners using dynamic weight scheduling
- Covers **32,947 users** and **50,683 tracks** — no artificial user or song caps

---

## Project structure

```
melodia/
├── app.py                      # Streamlit application (3 pages)
├── recommender.py              # Full hybrid engine + feature engineering
├── Music Info.csv              # 50,683 tracks with Spotify audio features
├── User Listening History.csv  # 448,523 raw play events across 43,636 users
└── README.md
```

---

## Quick start

### 1. Clone / download the project

Place all four files in the same folder:
```
app.py
recommender.py
Music Info.csv
User Listening History.csv
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn scipy
```

### 4. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. The model loads once on first run (~3 seconds), then all recommendations are instant.

---

## App pages

| Page | What it shows |
|---|---|
| **Overview** | Dataset stats, genre breakdown, user activity distribution, confusion matrix |
| **Users** | Searchable table of all 32,947 users — click any row to jump to their recommendations |
| **Recommendations** | Ranked tracks with match scores, NEW FIND / FAMILIAR badges, listening history, signal breakdown charts |

### Navigating to a user's recommendations

1. Go to the **Users** page
2. Use the segment buttons (All / Top 50 / Heavy / Moderate / Light) or search box to filter
3. Click any row in the table → select the user from the dropdown below the table
4. Press **▶ View recommendations**

---

## How the engine works

### Three signals, blended dynamically

```
final_score = w_CF × CF_score + w_Content × content_score + w_Novelty × novelty_score
```

| Signal | Method | What it captures |
|---|---|---|
| Collaborative filtering | TruncatedSVD (50 factors) on Borda-rank weighted matrix | Taste from users with similar behaviour |
| Content similarity | Cosine similarity on 80-dim audio vectors | Acoustic match (energy, valence, tempo, key, tags…) |
| Novelty | 1 − global popularity | Long-tail discovery |

### Dynamic weights by history depth

| User history | CF | Content | Novelty |
|---|---|---|---|
| 0–4 songs (cold start) | 0% | 80% | 20% |
| 5–19 songs | 30% | 55% | 15% |
| 20–49 songs | 55% | 35% | 10% |
| 50+ songs | 70% | 22% | 8% |

### MMR diversity re-ranking

After scoring, **Maximal Marginal Relevance** (Carbonell & Goldstein, 1998) re-orders the final list to prevent all results clustering around one artist or sub-genre:

```
MMR(s) = 0.70 × relevance(s) − 0.30 × max_similarity_to_already_selected
```

### Feature engineering highlights

- **Musical key** — encoded as sin/cos on the circle of fifths (avoids the B ≈ C distance error of linear encoding)
- **Year** — bucketed by decade + sin/cos (prevents 122× scale mismatch with audio features)
- **Genre** — 56% of tracks had null genre; 19,408 labels recovered from the `tags` column
- **Tags** — top-50 tags multi-hot encoded (adds mood/style signal: "chillout", "female_vocalists", etc.)
- **Total feature vector** — 80 dimensions per track

### Interaction weighting

Raw play counts use **Borda-rank weighting** rather than log(1 + playcount):

```
borda(user, song) = (n_songs − rank) / n_songs
```

Then z-score normalized per user so all users are on the same scale regardless of listening intensity.

---

## Evaluation results

| Metric | Popularity baseline | Hybrid SVD-CF | Improvement |
|---|---|---|---|
| Precision@10 | 0.035 | **0.4775** | +1,264% |
| NDCG@10 | 0.052 | **0.8872** | +1,609% |
| Novelty@10 | 3.647 | **4.380** | +20.1% |

**Recommendation latency:** 0.06 – 0.16 seconds per user (12× faster than the original implementation).

---

## Dataset

| File | Contents |
|---|---|
| `Music Info.csv` | 50,683 tracks — title, artist, genre, year, Spotify audio features (danceability, energy, valence, tempo, acousticness, speechiness, instrumentalness, liveness, loudness, key, mode), tags |
| `User Listening History.csv` | 448,523 play events — user_id, track_id, playcount |

**After quality filtering** (≥ 5 plays per user, ≥ 5 plays per song):
- 32,947 users in the CF model
- 15,829 songs in the interaction matrix  
- 50,683 tracks available as recommendation candidates (full catalog)
- 416,986 interactions used for training

> **No listening history file?** If `User Listening History.csv` is missing, the system auto-generates synthetic listening data (1,200 users, genre-weighted) and saves it. The app still runs fully — just with synthetic users.

---

## Configuration

All tunable parameters are at the top of `recommender.py`:

```python
MIN_USER_PLAYS  = 5      # minimum total plays to include a user
MIN_SONG_PLAYS  = 5      # minimum total plays to include a song in CF
N_FACTORS       = 50     # SVD latent factors (higher = more quality, slower startup)
TOP_N_TAGS      = 50     # number of tag features in content vector
MMR_LAMBDA      = 0.30   # 0 = pure relevance, 1 = pure diversity
```

---

## Dependencies

```
streamlit
pandas
numpy
scikit-learn
scipy
```

All installable with a single `pip install streamlit pandas numpy scikit-learn scipy`.  
No GPU required. No API keys required. No database required.

---

**Key references:**
- Koren et al. (2009) — Matrix Factorization for Recommender Systems
- Carbonell & Goldstein (1998) — MMR for diversity-based reranking
- Schedl et al. (2018) — Current challenges in Music Recommender Systems
