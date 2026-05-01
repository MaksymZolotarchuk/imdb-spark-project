from pyspark.sql import SparkSession
from src.transformation import run_all_transformations
from src.load import run_writing

PROCESSED_DIR = "/Users/Study/3.2/VVD/processed_imdb"
RESULTS_DIR = "results"
SEPARATOR = "=" * 70

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("IMDB Transformation")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{SEPARATOR}")
    print("  IMDB — ЕТАП ТРАНСФОРМАЦІЇ")
    print(SEPARATOR)

    print("\n  Зчитування підготовлених parquet-таблиць...")
    df_basics      = spark.read.parquet(f"{PROCESSED_DIR}/title_basics")
    df_ratings     = spark.read.parquet(f"{PROCESSED_DIR}/title_ratings")
    df_names       = spark.read.parquet(f"{PROCESSED_DIR}/name_basics")
    df_crew        = spark.read.parquet(f"{PROCESSED_DIR}/title_crew")
    df_principals  = spark.read.parquet(f"{PROCESSED_DIR}/title_principals")
    print("  ✔ Дані зчитано")

    results = run_all_transformations(
        df_basics=df_basics,
        df_ratings=df_ratings,
        df_names=df_names,
        df_crew=df_crew,
        df_principals=df_principals,
    )

    run_writing(results, output_dir=RESULTS_DIR)

    spark.stop()
    print("\n  ✅ Етап трансформації завершено успішно!")
