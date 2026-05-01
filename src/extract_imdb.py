from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)
from pyspark.sql import functions as F

NAME_BASICS_SCHEMA = StructType([
    StructField("nconst", StringType(), True),
    StructField("primaryName", StringType(), True),
    StructField("birthYear", IntegerType(), True),
    StructField("deathYear", IntegerType(), True),
    StructField("primaryProfession", StringType(), True),
    StructField("knownForTitles", StringType(), True),
])

TITLE_BASICS_SCHEMA = StructType([
    StructField("tconst", StringType(), True),
    StructField("titleType", StringType(), True),
    StructField("primaryTitle", StringType(), True),
    StructField("originalTitle", StringType(), True),
    StructField("isAdult", IntegerType(), True),
    StructField("startYear", IntegerType(), True),
    StructField("endYear", IntegerType(), True),
    StructField("runtimeMinutes", IntegerType(), True),
    StructField("genres", StringType(), True),
])

TITLE_CREW_SCHEMA = StructType([
    StructField("tconst", StringType(), True),
    StructField("directors", StringType(), True),
    StructField("writers", StringType(), True),
])

TITLE_PRINCIPALS_SCHEMA = StructType([
    StructField("tconst", StringType(), True),
    StructField("ordering", IntegerType(), True),
    StructField("nconst", StringType(), True),
    StructField("category", StringType(), True),
    StructField("job", StringType(), True),
    StructField("characters", StringType(), True),
])

TITLE_RATINGS_SCHEMA = StructType([
    StructField("tconst", StringType(), True),
    StructField("averageRating", DoubleType(), True),
    StructField("numVotes", IntegerType(), True),
])

def read_imdb_tsv(spark: SparkSession, path: str, schema: StructType) -> DataFrame:

    return (
        spark.read
        .option("sep", "\t")
        .option("header", "true")
        .option("nullValue", "\\N")
        .schema(schema)
        .csv(path)
    )

def extract_title_basics(spark: SparkSession, dataset_dir: str) -> DataFrame:
    df = read_imdb_tsv(spark, f"{dataset_dir}/title.basics.tsv", TITLE_BASICS_SCHEMA)
    _validate_df(df, "title.basics")
    return df

def extract_title_ratings(spark: SparkSession, dataset_dir: str) -> DataFrame:
    df = read_imdb_tsv(spark, f"{dataset_dir}/title.ratings.tsv", TITLE_RATINGS_SCHEMA)
    _validate_df(df, "title.ratings")
    return df

def extract_name_basics(spark: SparkSession, dataset_dir: str) -> DataFrame:
    df = read_imdb_tsv(spark, f"{dataset_dir}/name.basics.tsv", NAME_BASICS_SCHEMA)
    _validate_df(df, "name.basics")
    return df

def extract_title_crew(spark: SparkSession, dataset_dir: str) -> DataFrame:
    df = read_imdb_tsv(spark, f"{dataset_dir}/title.crew.tsv", TITLE_CREW_SCHEMA)
    _validate_df(df, "title.crew")
    return df

def extract_title_principals(spark: SparkSession, dataset_dir: str) -> DataFrame:
    df = read_imdb_tsv(spark, f"{dataset_dir}/title.principals.tsv", TITLE_PRINCIPALS_SCHEMA)
    _validate_df(df, "title.principals")
    return df

def _validate_df(df: DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    df.printSchema()
    print("rows =", df.count())
    df.show(5, truncate=False)
    if df.limit(1).count() == 0:
        raise ValueError(f"{name}: DataFrame порожній — перевір шлях/файл")
