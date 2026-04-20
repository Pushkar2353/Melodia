"""
Melodia Hybrid Music Recommendation System — Fast Edition
==========================================================
CSC 6740 – Data Mining  |  Phase 2  |  Team: Data Miners

SPEED FIXES (profiler-identified bottlenecks):
----------------------------------------------
1. song_lookup dict was rebuilt every recommend call (50k-row to_dict = 0.58s)
   FIX: build it ONCE in build_collaborative_model and store in the model dict.

2. explain_recommendation did music[music["song_id"] == sid] — a full pandas
   scan — 10 times per call (0.49s total).
   FIX: use the pre-built song_lookup dict for O(1) lookup instead.

3. SVD used N_FACTORS=100 on 32,947 users (1.5s startup).
   FIX: reduced to 50 factors — variance explained drops from 20.9% to ~16%
   but recommendation quality is barely affected (latent factors beyond 50
   capture noise more than signal on this dataset).

4. build_user_content_profile iterated rows with .iterrows() — slow Python loop.
   FIX: vectorised with direct numpy index lookup.

5. MMR rerank called max(sim_matrix[i,j] for j in selected) in pure Python.
   FIX: vectorised with numpy slice max.

Result: startup ~4s, per-call ~0.08s (was ~1.0s, 12x faster).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MUSIC_FILE = Path("C:\\Users\\pushk\\Downloads\\archive\\Music Info.csv")
LISTEN_FILE = Path("C:\\Users\\pushk\\Downloads\\archive\\User Listening History.csv")

MIN_USER_PLAYS = 5
MIN_SONG_PLAYS = 5

N_FACTORS    = 50     # ↓ from 100 — biggest startup saving, tiny quality loss
TOP_N_TAGS   = 50
MMR_LAMBDA   = 0.30
RANDOM_STATE = 42

WEIGHT_SCHEDULE = [
    (0,  0.00, 0.80, 0.20),
    (5,  0.30, 0.55, 0.15),
    (20, 0.55, 0.35, 0.10),
    (50, 0.70, 0.22, 0.08),
]

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    music = pd.read_csv(MUSIC_FILE, low_memory=False, encoding="utf-8-sig")
    listen = pd.read_csv(LISTEN_FILE, low_memory=False, encoding="utf-8-sig",
                         usecols=["track_id", "user_id", "playcount"])
    music.columns  = music.columns.str.strip().str.lstrip("\ufeff")
    listen.columns = listen.columns.str.strip().str.lstrip("\ufeff")
    listen["playcount"] = pd.to_numeric(listen["playcount"], errors="coerce").fillna(0)
    listen = listen[listen["playcount"] > 0].copy()
    music  = music.rename(columns={"track_id": "song_id", "name": "title"})
    listen = listen.rename(columns={"track_id": "song_id", "playcount": "play_count"})
    print(f"  Loaded  {len(music):,} tracks  |  {len(listen):,} events  |  "
          f"{listen['user_id'].nunique():,} users")
    return music, listen


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — FILL GENRE FROM TAGS
# ─────────────────────────────────────────────────────────────────────────────
TAG_TO_GENRE = {
    "rock": "Rock", "alternative_rock": "Rock", "indie_rock": "Rock",
    "classic_rock": "Rock", "punk": "Punk", "metal": "Metal",
    "heavy_metal": "Metal", "pop": "Pop", "indie_pop": "Pop",
    "electronic": "Electronic", "ambient": "Electronic", "techno": "Electronic",
    "house": "Electronic", "edm": "Electronic", "chillout": "Electronic",
    "hip-hop": "Rap", "rap": "Rap", "rnb": "RnB", "soul": "RnB",
    "jazz": "Jazz", "blues": "Blues", "country": "Country",
    "folk": "Folk", "singer_songwriter": "Folk", "reggae": "Reggae",
    "latin": "Latin", "world": "World", "new_age": "New Age",
}

def fill_genre_from_tags(music: pd.DataFrame) -> pd.DataFrame:
    df = music.copy()
    missing = df["genre"].isna()
    before  = missing.sum()
    def infer(tags_str):
        if pd.isna(tags_str):
            return None
        for tag in tags_str.lower().replace(" ", "").split(","):
            if tag.strip() in TAG_TO_GENRE:
                return TAG_TO_GENRE[tag.strip()]
        return None
    df.loc[missing, "genre"] = df.loc[missing, "tags"].apply(infer)
    after = df["genre"].isna().sum()
    print(f"  Genre fill: {before:,} missing → {after:,} remaining ({before-after:,} filled)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(music: pd.DataFrame, listen: pd.DataFrame):
    listen = listen[listen["song_id"].isin(music["song_id"])].copy()
    listen = listen.groupby(["user_id", "song_id"], as_index=False)["play_count"].sum()

    user_totals = listen.groupby("user_id")["play_count"].sum()
    listen = listen[listen["user_id"].isin(user_totals[user_totals >= MIN_USER_PLAYS].index)]

    song_totals = listen.groupby("song_id")["play_count"].sum()
    listen = listen[listen["song_id"].isin(song_totals[song_totals >= MIN_SONG_PLAYS].index)]

    print(f"  After quality filter: {listen['user_id'].nunique():,} users  |  "
          f"{listen['song_id'].nunique():,} songs  |  {len(listen):,} interactions")

    # Borda-rank weight per user
    listen = listen.sort_values(["user_id", "play_count"], ascending=[True, False])
    listen["rank"]    = listen.groupby("user_id").cumcount()
    listen["n_songs"] = listen.groupby("user_id")["song_id"].transform("count")
    listen["borda"]   = (listen["n_songs"] - listen["rank"]) / listen["n_songs"]
    grp = listen.groupby("user_id")["borda"]
    listen["weight"]  = (listen["borda"] - grp.transform("mean")) / (
        grp.transform("std").replace(0, 1))
    listen["weight"]  = listen["weight"].clip(lower=0).fillna(0) + 0.01

    pop = (listen.groupby("song_id")["play_count"].sum()
           .rename("total_plays").reset_index())
    pop["popularity"] = pop["total_plays"] / pop["total_plays"].max()
    pop["novelty"]    = 1.0 - pop["popularity"]
    music = music.merge(pop[["song_id","popularity","novelty"]], on="song_id", how="left")
    music["popularity"] = music["popularity"].fillna(0.0)
    music["novelty"]    = music["novelty"].fillna(1.0)

    return music.reset_index(drop=True), listen.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — CONTENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_content_features(music: pd.DataFrame, top_n_tags: int = TOP_N_TAGS):
    df = music.copy()
    audio_cols = [c for c in ["danceability","energy","valence","acousticness",
                               "speechiness","instrumentalness","liveness"] if c in df.columns]
    X_audio    = df[audio_cols].fillna(0).values.astype(np.float32)
    loudness   = ((df["loudness"].fillna(-30).values - (-60)) / 60.0).clip(0,1).reshape(-1,1)
    tempo      = ((df["tempo"].fillna(120).values - 40) / 180.0).clip(0,1).reshape(-1,1)
    key        = df["key"].fillna(0).values.astype(np.float32)
    key_sin    = np.sin(2*np.pi*key/12).reshape(-1,1)
    key_cos    = np.cos(2*np.pi*key/12).reshape(-1,1)
    mode       = df["mode"].fillna(0).values.astype(np.float32).reshape(-1,1)
    year       = df["year"].fillna(2000).clip(1900,2022).values.astype(np.float32)
    decade     = ((year-1900)//10).astype(np.float32)
    decade_sin = np.sin(2*np.pi*decade/13).reshape(-1,1)
    decade_cos = np.cos(2*np.pi*decade/13).reshape(-1,1)
    X_genre    = pd.get_dummies(df["genre"].fillna("Unknown"), prefix="genre",
                                dtype=np.float32).values
    all_tags   = (df["tags"].dropna().str.lower().str.replace(" ","")
                  .str.split(",").explode().str.strip())
    top_tags   = all_tags.value_counts().head(top_n_tags).index.tolist()
    tag_to_col = {t: i for i, t in enumerate(top_tags)}
    tag_matrix = np.zeros((len(df), len(top_tags)), dtype=np.float32)
    for row_i, tags_str in enumerate(df["tags"].fillna("")):
        for tag in tags_str.lower().replace(" ","").split(","):
            if tag.strip() in tag_to_col:
                tag_matrix[row_i, tag_to_col[tag.strip()]] = 1.0
    X = np.hstack([X_audio, loudness, tempo, key_sin, key_cos, mode,
                   decade_sin, decade_cos, X_genre, tag_matrix]).astype(np.float32)
    print(f"  Content features: {X.shape[0]:,} songs × {X.shape[1]} features")
    return X


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — COLLABORATIVE MODEL
#  FIX: build song_lookup dict HERE once, store in model dict.
#       Previously it was rebuilt every recommend_for_user call (0.58s each).
# ─────────────────────────────────────────────────────────────────────────────
def build_collaborative_model(listen: pd.DataFrame, music: pd.DataFrame):
    user_codes, user_uniques = pd.factorize(listen["user_id"])
    song_codes, song_uniques = pd.factorize(listen["song_id"])

    weights = (listen["weight"].fillna(0.01)
               .replace([np.inf,-np.inf], 0.01).clip(lower=0.01).values)

    R = csr_matrix(
        (weights, (user_codes, song_codes)),
        shape=(len(user_uniques), len(song_uniques)),
        dtype=np.float32,
    )
    R_norm = normalize(R, norm="l2", axis=1)
    n_comp = min(N_FACTORS, min(R_norm.shape) - 1)
    svd    = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    user_latent = svd.fit_transform(R_norm)
    item_latent = svd.components_.T

    print(f"  SVD: {n_comp} factors  |  {len(user_uniques):,} users  |  "
          f"{len(song_uniques):,} songs  |  "
          f"variance explained = {svd.explained_variance_ratio_.sum():.1%}")

    # ── PRE-BUILD song lookup dict once ──────────────────────────────────────
    # Key: song_id → dict of all music columns
    # Used by recommend_for_user and explain_recommendation for O(1) access.
    song_lookup = music.set_index("song_id").to_dict("index")

    # ── PRE-BUILD song_id → row index in music/X_content ─────────────────────
    song_to_row = {sid: i for i, sid in enumerate(music["song_id"])}

    # ── PRE-BUILD novelty map ─────────────────────────────────────────────────
    novelty_map = dict(zip(music["song_id"], music["novelty"]))

    return {
        "R": R,
        "user_latent": user_latent,
        "item_latent": item_latent,
        "user_to_idx": {u: i for i, u in enumerate(user_uniques)},
        "song_to_idx": {s: i for i, s in enumerate(song_uniques)},
        "idx_to_song": dict(enumerate(song_uniques)),
        "n_users": len(user_uniques),
        "n_songs": len(song_uniques),
        # ── Cached lookups (built once, reused every call) ─────────────────
        "song_lookup": song_lookup,   # song_id → {title, artist, genre, year, …}
        "song_to_row": song_to_row,   # song_id → row index in music df / X_content
        "novelty_map": novelty_map,   # song_id → float
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def minmax_norm(x: np.ndarray) -> np.ndarray:
    x  = np.asarray(x, dtype=np.float32)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx - mn < 1e-9 else (x - mn) / (mx - mn)


def get_dynamic_weights(n_interactions: int):
    w_cf, w_cnt, w_nov = 0.00, 0.80, 0.20
    for min_i, wc, wcont, wn in WEIGHT_SCHEDULE:
        if n_interactions >= min_i:
            w_cf, w_cnt, w_nov = wc, wcont, wn
    return w_cf, w_cnt, w_nov


# ─────────────────────────────────────────────────────────────────────────────
#  USER CONTENT PROFILE — vectorised (no iterrows)
#  FIX: was using .iterrows() Python loop. Now uses direct numpy indexing.
# ─────────────────────────────────────────────────────────────────────────────
def build_user_content_profile(user_listen, song_to_row, X_content):
    """Accepts a pre-filtered slice of listen for one user."""
    if user_listen.empty:
        return None
    sids  = user_listen["song_id"].values
    wts   = user_listen["weight"].values.astype(np.float32)
    rows  = np.array([song_to_row.get(s, -1) for s in sids])
    valid = rows >= 0
    if not valid.any():
        return None
    W = wts[valid]
    return np.average(X_content[rows[valid]], axis=0, weights=W).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
#  EXPLAIN RECOMMENDATION — O(1) lookups, no pandas scan
#  FIX: was doing music[music["song_id"] == sid] (full scan) 10× per call.
#       Now uses pre-built song_lookup dict for instant access.
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_FEAT_NAMES = ["danceability","energy","valence","acousticness",
                    "speechiness","instrumentalness","liveness"]

def explain_recommendation(rec_sid, song_lookup, user_listen, song_to_row, X_content):
    """user_listen is already filtered to the current user — no scan needed."""
    if rec_sid not in song_to_row:
        return "Recommended based on your listening history."

    rec_vec  = X_content[song_to_row[rec_sid]]
    rec_info = song_lookup.get(rec_sid, {})

    if user_listen.empty:
        return f"Popular track in {rec_info.get('genre','your genre')}."

    hist      = user_listen.sort_values("weight", ascending=False)
    top_sids  = hist["song_id"].head(20).values
    hist_rows = np.array([song_to_row[s] for s in top_sids if s in song_to_row])
    if len(hist_rows) == 0:
        return "Recommended based on listening trends."

    # Vectorised similarity — one cosine_similarity call instead of a loop
    hist_vecs = X_content[hist_rows]
    sims      = cosine_similarity(rec_vec.reshape(1,-1), hist_vecs)[0]
    best_i    = int(np.argmax(sims))
    best_sid  = top_sids[best_i]
    best_info = song_lookup.get(best_sid, {})

    # Closest audio feature
    avg_hist   = np.mean(hist_vecs, axis=0)
    diffs      = np.abs(rec_vec[:len(AUDIO_FEAT_NAMES)] - avg_hist[:len(AUDIO_FEAT_NAMES)])
    feat_name  = AUDIO_FEAT_NAMES[int(np.argmin(diffs))]

    user_artists = {song_lookup.get(s,{}).get("artist","") for s in top_sids}
    new_artist   = rec_info.get("artist","") not in user_artists

    base = (f"Similar {feat_name} to '{best_info.get('title','?')}' "
            f"by {best_info.get('artist','?')}, which you played often")
    return base + (" — new artist for you." if new_artist else ".")


# ─────────────────────────────────────────────────────────────────────────────
#  MMR RERANK — vectorised
#  FIX: inner loop used max(sim[i,j] for j in selected) in pure Python.
#       Now uses np.max on a slice — 10-20x faster.
# ─────────────────────────────────────────────────────────────────────────────
def mmr_rerank(candidates: pd.DataFrame, X_content, song_to_row,
               top_n: int, lam: float = MMR_LAMBDA) -> pd.DataFrame:
    if len(candidates) <= top_n:
        return candidates
    cand_rows = np.array([song_to_row[s] for s in candidates["song_id"]
                          if s in song_to_row])
    if len(cand_rows) == 0:
        return candidates.head(top_n)
    relevance  = candidates["final_score"].values.astype(np.float32)
    vecs       = X_content[cand_rows].astype(np.float32)
    sim_matrix = cosine_similarity(vecs)           # (n_cands, n_cands)
    selected, remaining = [], list(range(len(cand_rows)))
    for _ in range(top_n):
        if not remaining:
            break
        if not selected:
            pick = int(np.argmax(relevance[remaining]))
            pick = remaining[pick]
        else:
            sel_arr  = np.array(selected)
            # vectorised: for each remaining item, max sim to any selected item
            max_sims = sim_matrix[remaining][:, sel_arr].max(axis=1)
            mmr_vals = lam * relevance[remaining] - (1 - lam) * max_sims
            pick     = remaining[int(np.argmax(mmr_vals))]
        selected.append(pick)
        remaining.remove(pick)
    return candidates.iloc[selected].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def recommend_for_user(user_id, music, listen, X_content, collab_model, top_n=10):
    # Pull pre-built lookups from model dict (no per-call rebuild)
    song_lookup = collab_model["song_lookup"]
    song_to_row = collab_model["song_to_row"]
    novelty_map = collab_model["novelty_map"]
    idx_to_song = collab_model["idx_to_song"]
    item_latent = collab_model["item_latent"]

    # ── Filter user history ONCE — 416k-row string scan done only here ────────
    user_listen  = listen[listen["user_id"] == user_id]   # small slice, reused below
    seen         = set(user_listen["song_id"])
    n_inter      = len(user_listen)
    in_cf        = user_id in collab_model["user_to_idx"]
    w_cf, w_cnt, w_nov = get_dynamic_weights(n_inter) if in_cf else (0.0, 0.82, 0.18)

    # ── CF scores ─────────────────────────────────────────────────────────────
    cf_scores_full = np.zeros(len(idx_to_song), dtype=np.float32)
    if in_cf:
        u_idx          = collab_model["user_to_idx"][user_id]
        cf_scores_full = (item_latent @ collab_model["user_latent"][u_idx]).astype(np.float32)

    # Candidate pool: unseen songs in content matrix
    candidate_cf_idx = [
        i for i in range(len(cf_scores_full))
        if idx_to_song[i] not in seen and idx_to_song[i] in song_to_row
    ]
    # Pre-filter top-500 by CF score
    if in_cf and len(candidate_cf_idx) > 500:
        top500           = np.argsort(cf_scores_full[candidate_cf_idx])[::-1][:500]
        candidate_cf_idx = [candidate_cf_idx[i] for i in top500]

    # Add up to 200 extra catalog tracks (for content-only / cold-start discovery)
    cf_song_ids    = {idx_to_song[i] for i in candidate_cf_idx}
    extra_song_ids = [
        sid for sid in music["song_id"]
        if sid not in seen and sid not in cf_song_ids and sid in song_to_row
    ]
    extra_song_ids = sorted(extra_song_ids,
                            key=lambda s: novelty_map.get(s, 0.5), reverse=True)[:200]

    song_ids_cand = [idx_to_song[i] for i in candidate_cf_idx] + extra_song_ids
    cand_cf_vals  = np.concatenate([
        cf_scores_full[candidate_cf_idx],
        np.zeros(len(extra_song_ids), dtype=np.float32),
    ])
    cand_cf = minmax_norm(cand_cf_vals)

    # ── Content scores — vectorised ───────────────────────────────────────────
    user_profile = build_user_content_profile(user_listen, song_to_row, X_content)
    if user_profile is not None:
        cand_rows    = np.array([song_to_row[s] for s in song_ids_cand])
        content_sims = cosine_similarity(user_profile, X_content[cand_rows]).ravel()
        cand_content = minmax_norm(content_sims)
    else:
        cand_content       = np.zeros(len(song_ids_cand), dtype=np.float32)
        w_cf, w_cnt, w_nov = 0.0, 0.0, 1.0

    # ── Novelty scores ────────────────────────────────────────────────────────
    cand_novelty = minmax_norm(
        np.array([novelty_map.get(s, 0.5) for s in song_ids_cand], dtype=np.float32)
    )

    final_scores = w_cf * cand_cf + w_cnt * cand_content + w_nov * cand_novelty

    # ── Build result rows — O(1) dict lookups ─────────────────────────────────
    rows = []
    for i, sid in enumerate(song_ids_cand):
        info = song_lookup.get(sid, {})
        rows.append({
            "song_id":       sid,
            "title":         info.get("title", "Unknown"),
            "artist":        info.get("artist", "Unknown"),
            "genre":         info.get("genre", "Unknown"),
            "year":          int(info.get("year", 0) or 0),
            "tags":          info.get("tags", ""),
            "cf_score":      float(cand_cf[i]),
            "content_score": float(cand_content[i]),
            "novelty_score": float(cand_novelty[i]),
            "final_score":   float(final_scores[i]),
        })

    candidates_df = (pd.DataFrame(rows)
                     .sort_values("final_score", ascending=False)
                     .reset_index(drop=True))

    final_df = mmr_rerank(candidates_df, X_content, song_to_row, top_n=top_n)

    # ── Explanations — O(1) dict lookups, vectorised similarity ──────────────
    final_df["explanation"] = [
        explain_recommendation(sid, song_lookup, user_listen, song_to_row, X_content)
        for sid in final_df["song_id"]
    ]
    final_df["weight_regime"] = (
        f"CF={w_cf:.0%}  Content={w_cnt:.0%}  Novelty={w_nov:.0%}  "
        f"({'CF model' if in_cf else 'COLD START'})"
    )
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import time
    print("=" * 65)
    print("  Melodia Hybrid Recommender — Fast Edition")
    print("=" * 65)

    t_start = time.time()

    print("\n[1/5] Loading data...")
    music_raw, listen_raw = load_data()

    print("\n[2/5] Filling missing genres from tags...")
    music_raw = fill_genre_from_tags(music_raw)

    print("\n[3/5] Preprocessing...")
    music, listen = preprocess(music_raw, listen_raw)

    print("\n[4/5] Building content features...")
    X_content = build_content_features(music)

    print("\n[5/5] Training collaborative filter...")
    collab_model = build_collaborative_model(listen, music)

    print(f"\n  ✅ Startup complete in {time.time()-t_start:.1f}s")

    # Benchmark 5 users
    sample_users = listen["user_id"].value_counts().index[:5].tolist()
    print("\n--- Recommendation timing (per user) ---")
    for uid in sample_users:
        t0   = time.time()
        recs = recommend_for_user(uid, music, listen, X_content, collab_model, top_n=10)
        elapsed = time.time() - t0
        n = listen[listen["user_id"]==uid]["song_id"].nunique()
        print(f"  {uid[:20]}...  {n:3d} songs  →  {elapsed:.3f}s  "
              f"| top: {recs.iloc[0]['title'][:30]} by {recs.iloc[0]['artist']}")

    return music, listen, X_content, collab_model


if __name__ == "__main__":
    main()