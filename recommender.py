"""
CodSoft ML Internship - Task 4: Recommendation System
--------------------------------------------------------
A simple CONTENT-BASED movie recommendation system.

Approach:
    Each movie is described by its genres/keywords (its "content").
    We convert these descriptions into TF-IDF vectors, then use
    cosine similarity to find movies most similar to a movie the
    user already likes -- no user rating history required.

Dataset:
    Uses the TMDB 5000 Movies dataset from Kaggle (~4800 movies):
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
    Download "tmdb_5000_movies.csv" and place it next to this script
    (or pass its path with --data).

    If the CSV isn't found, the script automatically falls back to a
    small 15-movie built-in sample so it still runs out of the box.

Usage:
    python recommender.py --movie "Inception"
    python recommender.py --movie "Inception" --top 10
    python recommender.py --list          (see all available movies)
    python recommender.py --data tmdb_5000_movies.csv --movie "Avatar"

Requirements:
    pip install pandas scikit-learn
"""

import argparse
import ast
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATA_PATH = "tmdb_5000_movies.csv"

# --------------------------------------------------------------------------
# Fallback sample dataset, used only if the Kaggle CSV isn't found.
# --------------------------------------------------------------------------
SAMPLE_MOVIES = [
    {"title": "Inception", "genres": "Sci-Fi Thriller Action",
     "keywords": "dream heist subconscious mind-bending"},
    {"title": "Interstellar", "genres": "Sci-Fi Drama Adventure",
     "keywords": "space time travel wormhole survival"},
    {"title": "The Dark Knight", "genres": "Action Crime Drama",
     "keywords": "superhero vigilante crime city"},
    {"title": "The Prestige", "genres": "Drama Mystery Thriller",
     "keywords": "magic rivalry obsession illusion"},
    {"title": "Memento", "genres": "Mystery Thriller",
     "keywords": "memory loss revenge nonlinear"},
    {"title": "The Matrix", "genres": "Sci-Fi Action",
     "keywords": "simulation virtual reality rebellion"},
    {"title": "Titanic", "genres": "Romance Drama",
     "keywords": "ship disaster love tragedy"},
    {"title": "The Notebook", "genres": "Romance Drama",
     "keywords": "love letters memory relationship"},
    {"title": "La La Land", "genres": "Romance Musical Drama",
     "keywords": "jazz dreams love hollywood"},
    {"title": "John Wick", "genres": "Action Thriller Crime",
     "keywords": "assassin revenge gun-fu underworld"},
    {"title": "Mad Max: Fury Road", "genres": "Action Adventure Sci-Fi",
     "keywords": "wasteland chase survival dystopia"},
    {"title": "Gravity", "genres": "Sci-Fi Drama Thriller",
     "keywords": "space astronaut survival isolation"},
    {"title": "The Shawshank Redemption", "genres": "Drama",
     "keywords": "prison friendship hope redemption"},
    {"title": "Forrest Gump", "genres": "Drama Romance Comedy",
     "keywords": "life journey history destiny"},
    {"title": "Get Out", "genres": "Horror Thriller Mystery",
     "keywords": "psychological racism twist suspense"},
]


def _parse_json_list_column(value):
    """
    TMDB's CSV stores genres/keywords as stringified lists of dicts, e.g.
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    This pulls out just the 'name' values as a space-separated string.
    """
    if pd.isna(value):
        return ""
    try:
        items = ast.literal_eval(value)
        return " ".join(item["name"] for item in items)
    except (ValueError, SyntaxError, TypeError):
        return ""


def build_dataframe(data_path=DEFAULT_DATA_PATH):
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path} ...")
        raw = pd.read_csv(data_path)

        df = pd.DataFrame()
        df["title"] = raw["title"]
        df["genres"] = raw["genres"].apply(_parse_json_list_column)
        df["keywords"] = raw["keywords"].apply(_parse_json_list_column) if "keywords" in raw.columns else ""
        overview = raw["overview"].fillna("") if "overview" in raw.columns else ""

        df["content"] = df["genres"] + " " + df["keywords"] + " " + overview
        df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"]).reset_index(drop=True)
        print(f"Loaded {len(df)} movies.")
        return df

    print(f"'{data_path}' not found -- using the built-in 15-movie sample dataset instead.")
    print("Download tmdb_5000_movies.csv from Kaggle and place it next to this script to use the full dataset.")
    df = pd.DataFrame(SAMPLE_MOVIES)
    df["content"] = df["genres"] + " " + df["keywords"]
    return df


def build_similarity_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


def recommend(movie_title, df, similarity, top_n=5):
    matches = df[df["title"].str.lower() == movie_title.lower()]
    if matches.empty:
        print(f"'{movie_title}' not found in the dataset.")
        print("Use --list to see available titles.")
        return

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    # sort by similarity score, skip the movie itself (always similarity 1.0 with itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    print(f"\nBecause you liked '{df.loc[idx, 'title']}', you might also like:\n")
    for rank, (movie_idx, score) in enumerate(scores, start=1):
        title = df.loc[movie_idx, "title"]
        genres = df.loc[movie_idx, "genres"]
        print(f"  {rank}. {title}  (genres: {genres}, similarity: {score:.2f})")


def list_movies(df, limit=30):
    print(f"\n{len(df)} movies available. Showing first {min(limit, len(df))}:\n")
    for title in df["title"].head(limit):
        print(f"  - {title}")
    if len(df) > limit:
        print(f"  ... and {len(df) - limit} more. Use --movie \"Title\" to search directly.")


def main():
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("--movie", type=str, help="Title of a movie you like")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations (default 5)")
    parser.add_argument("--list", action="store_true", help="List all available movie titles")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                         help=f"Path to tmdb_5000_movies.csv (default: {DEFAULT_DATA_PATH})")
    args = parser.parse_args()

    df = build_dataframe(args.data)

    if args.list:
        list_movies(df)
        return

    if not args.movie:
        print("Please specify a movie with --movie \"Title\", or use --list to see options.")
        return

    similarity = build_similarity_matrix(df)
    recommend(args.movie, df, similarity, top_n=args.top)


if __name__ == "__main__":
    main()
