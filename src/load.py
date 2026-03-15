"""
Етап запису результатів для IMDb dataset.
Зберігає відповіді на 6 бізнес-питань у файли .csv
"""

import os

SEPARATOR = "=" * 70
OUTPUT_DIR = "results"


def section(title: str):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def save_result(df, name: str, output_dir: str = OUTPUT_DIR):
    """
    Зберігає DataFrame у єдиний .csv файл.
    coalesce(1) об'єднує всі партиції в один файл перед записом.
    """
    path = os.path.join(output_dir, name)

    (
        df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("encoding", "UTF-8")
        .csv(path)
    )

    print(f"  ✔ Збережено: {path}/")


def run_writing(results: dict, output_dir: str = OUTPUT_DIR):
    section("ЕТАП ЗАПИСУ РЕЗУЛЬТАТІВ")
    print(f"  Директорія виводу: {output_dir}/\n")

    os.makedirs(output_dir, exist_ok=True)

    mapping = {
        "q1_top_rated_movies":      "q1_top_rated_movies",
        "q2_top_genres_high_rated": "q2_top_genres_high_rated",
        "q3_top_years_by_movie_count": "q3_top_years_by_movie_count",
        "q4_top3_movies_per_genre": "q4_top3_movies_per_genre",
        "q5_top_directors_by_titles": "q5_top_directors_by_titles",
        "q6_best_actors_by_avg_rating": "q6_best_actors_by_avg_rating",
    }

    for key, folder_name in mapping.items():
        df = results.get(key)
        if df is not None:
            save_result(df, folder_name, output_dir)
        else:
            print(f"  ⚠ Пропущено (не знайдено): {key}")

    print(f"\n  ✅ Усі результати збережено у папку '{output_dir}/'")
