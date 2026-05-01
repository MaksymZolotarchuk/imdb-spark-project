from pyspark.sql import SparkSession

from src.extract_imdb import (
    extract_title_basics,
    extract_title_ratings,
    extract_name_basics,
    extract_title_crew,
    extract_title_principals,
)
from src.preprocessing import (
    dataset_info,
    numeric_statistics,
    cast_types,
    remove_uninformative_columns,
    check_missing,
    remove_duplicates,
)

DATASET_DIR = "/Users/Study/3.2/VVD/Datasets"
OUTPUT_DIR = "/Users/Study/3.2/VVD/processed_imdb"
SEPARATOR = "=" * 70


def preprocess(df, name: str):
    print(f"\n{'#' * 70}")
    print(f"#  ОБРОБКА: {name}")
    print(f"{'#' * 70}")

    dataset_info(df, name)
    df = cast_types(df, name)
    numeric_statistics(df, name)
    df = remove_uninformative_columns(df, name)
    df = check_missing(df, name)

    if name == "title_principals":
        df = remove_duplicates(df, name, subset=["tconst", "ordering", "nconst"])
    elif name == "name_basics":
        df = remove_duplicates(df, name, subset=["nconst"])
    elif "tconst" in df.columns:
        df = remove_duplicates(df, name, subset=["tconst"])
    else:
        df = remove_duplicates(df, name)

    print(f"\n  ✅ {name} — обробку завершено. Рядків: {df.count():,} | Колонок: {len(df.columns)}")
    return df


def save_df(df, path: str):
    (
        df.write
        .mode("overwrite")
        .parquet(path)
    )
    print(f"  ✔ Збережено: {path}")


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("IMDB Preprocessing")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{SEPARATOR}")
    print("  IMDB — ПОПЕРЕДНЯ ОБРОБКА ДАНИХ (PySpark)")
    print(SEPARATOR)

    print("\n  Завантаження датасетів...")
    df_basics = extract_title_basics(spark, DATASET_DIR)
    df_ratings = extract_title_ratings(spark, DATASET_DIR)
    df_names = extract_name_basics(spark, DATASET_DIR)
    df_crew = extract_title_crew(spark, DATASET_DIR)
    df_principals = extract_title_principals(spark, DATASET_DIR)
    print("  ✔ Усі датасети завантажено")

    df_basics = preprocess(df_basics, "title_basics")
    df_ratings = preprocess(df_ratings, "title_ratings")
    df_names = preprocess(df_names, "name_basics")
    df_crew = preprocess(df_crew, "title_crew")
    df_principals = preprocess(df_principals, "title_principals")

    print(f"\n{SEPARATOR}")
    print("  ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print(SEPARATOR)

    save_df(df_basics, f"{OUTPUT_DIR}/title_basics")
    save_df(df_ratings, f"{OUTPUT_DIR}/title_ratings")
    save_df(df_names, f"{OUTPUT_DIR}/name_basics")
    save_df(df_crew, f"{OUTPUT_DIR}/title_crew")
    save_df(df_principals, f"{OUTPUT_DIR}/title_principals")

    print(f"\n{SEPARATOR}")
    print("  ПІДСУМОК ПОПЕРЕДНЬОЇ ОБРОБКИ")
    print(SEPARATOR)

    results = {
        "title_basics": df_basics,
        "title_ratings": df_ratings,
        "name_basics": df_names,
        "title_crew": df_crew,
        "title_principals": df_principals,
    }

    print(f"\n  {'Таблиця':<22} {'Рядків':>12} {'Колонок':>10}")
    print("  " + "-" * 46)
    for tname, df in results.items():
        print(f"  {tname:<22} {df.count():>12,} {len(df.columns):>10}")

    spark.stop()
    print("\n  ✅ Пайплайн завершено успішно!")
