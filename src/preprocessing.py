"""
src/preprocessing.py
====================
Попередня обробка IMDB датасетів на PySpark.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    FloatType,
    DoubleType,
    LongType,
    ShortType,
    StringType,
    BooleanType,
    ArrayType,
)

SEPARATOR = "=" * 70


def _section(title: str):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def _is_numeric_type(dtype) -> bool:
    return isinstance(dtype, (IntegerType, FloatType, DoubleType, LongType, ShortType))


# ─────────────────────────────────────────────────────────────────────
# КРОК 1 — Загальна інформація про датасет
# ─────────────────────────────────────────────────────────────────────
def dataset_info(df: DataFrame, name: str = "DataFrame") -> None:
    _section(f"КРОК 1 — ЗАГАЛЬНА ІНФОРМАЦІЯ: {name}")

    row_count = df.count()
    col_count = len(df.columns)

    print(f"\n  Таблиця  : {name}")
    print(f"  Рядків   : {row_count:,}")
    print(f"  Стовпців : {col_count}")

    print("\n  Схема:")
    df.printSchema()

    print("\n  Перші 5 рядків:")
    df.show(5, truncate=True)

    print("  Пропущені значення:")
    null_counts = df.select([
        F.round(F.mean(F.col(c).isNull().cast("double")) * 100, 2).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    for col_name, pct in null_counts.items():
        bar = "█" * int((pct or 0) // 5)
        print(f"    {col_name:<30} {pct:>6.2f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────
# КРОК 2 — Статистика числових ознак
# ─────────────────────────────────────────────────────────────────────
def numeric_statistics(df: DataFrame, name: str = "DataFrame") -> None:
    _section(f"КРОК 2 — ЧИСЛОВІ ОЗНАКИ: {name}")

    numeric_cols = [
        field.name for field in df.schema.fields
        if _is_numeric_type(field.dataType)
    ]

    if not numeric_cols:
        print("  ⚠ Числових колонок не знайдено.")
        return

    print(f"\n  Числові колонки: {numeric_cols}")
    print("\n  describe():")
    df.select(*numeric_cols).describe().show(truncate=False)

    print("  Квантилі (0.25 / 0.5 / 0.75 / 0.95):")
    for col_name in numeric_cols:
        try:
            quantiles = df.stat.approxQuantile(col_name, [0.25, 0.5, 0.75, 0.95], 0.01)
            if len(quantiles) == 4:
                print(
                    f"    {col_name:<25} "
                    f"Q1={quantiles[0]:.1f}  Med={quantiles[1]:.1f}  "
                    f"Q3={quantiles[2]:.1f}  P95={quantiles[3]:.1f}"
                )
        except Exception:
            print(f"    {col_name:<25} — не вдалося обчислити")


# ─────────────────────────────────────────────────────────────────────
# КРОК 3 — Приведення типів
# ─────────────────────────────────────────────────────────────────────
def cast_types(df: DataFrame, name: str = "DataFrame") -> DataFrame:
    _section(f"КРОК 3 — ПРИВЕДЕННЯ ТИПІВ: {name}")

    print("  Схема ДО приведення:")
    df.printSchema()

    # Замінюємо літеральний \N на null тільки в рядкових колонках
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(
                field.name,
                F.when(F.col(field.name) == r"\N", None).otherwise(F.col(field.name))
            )

    cols = set(df.columns)

    int_cols = {
        "startYear", "endYear", "runtimeMinutes",
        "birthYear", "deathYear", "ordering",
        "seasonNumber", "episodeNumber", "numVotes"
    } & cols

    for col_name in int_cols:
        df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))

    if "averageRating" in cols:
        df = df.withColumn("averageRating", F.col("averageRating").cast(FloatType()))

    if "isAdult" in cols:
        df = df.withColumn(
            "isAdult",
            F.when(F.col("isAdult").cast("string") == "1", F.lit(True))
             .when(F.col("isAdult").cast("string") == "0", F.lit(False))
             .otherwise(F.lit(None).cast(BooleanType()))
        )

    if "genres" in cols:
        df = df.withColumn(
            "genres",
            F.when(F.col("genres").isNull(), None)
             .otherwise(F.split(F.col("genres"), ","))
        )

    if "knownForTitles" in cols:
        df = df.withColumn(
            "knownForTitles",
            F.when(F.col("knownForTitles").isNull(), None)
             .otherwise(F.split(F.col("knownForTitles"), ","))
        )

    if "primaryProfession" in cols:
        df = df.withColumn(
            "primaryProfession",
            F.when(F.col("primaryProfession").isNull(), None)
             .otherwise(F.split(F.col("primaryProfession"), ","))
        )

    for col_name in {"directors", "writers"} & cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), None)
             .otherwise(F.split(F.col(col_name), ","))
        )

    print("\n  Схема ПІСЛЯ приведення:")
    df.printSchema()
    return df


# ─────────────────────────────────────────────────────────────────────
# КРОК 4 — Аналіз інформативності та вилучення ознак
# ─────────────────────────────────────────────────────────────────────
def remove_uninformative_columns(
    df: DataFrame,
    name: str = "DataFrame",
    null_threshold: float = 0.85,
) -> DataFrame:
    _section(f"КРОК 4 — ІНФОРМАТИВНІСТЬ ОЗНАК: {name}")

    total = df.count()
    cols_to_drop = []

    null_fracs = df.select([
        F.mean(F.col(c).isNull().cast("double")).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    print(f"\n  Частка null по колонках (поріг = {null_threshold:.0%}):")
    for col_name, frac in sorted(null_fracs.items(), key=lambda x: -x[1]):
        flag = " ← ВИЛУЧИТИ" if frac > null_threshold else ""
        print(f"    {col_name:<30} {frac * 100:>6.2f}%{flag}")

    print("\n  Перевірка константних колонок:")
    for col_name in df.columns:
        n_unique = df.select(col_name).distinct().count()
        if n_unique <= 1:
            print(f"    {col_name:<30} унікальних: {n_unique} ← ВИЛУЧИТИ (константа)")
            cols_to_drop.append(col_name)
        else:
            print(f"    {col_name:<30} унікальних: {n_unique:,}")

    # Спеціальні правила для IMDb
    if name == "title_basics":
        if "endYear" in df.columns and null_fracs.get("endYear", 0) > null_threshold:
            cols_to_drop.append("endYear")

        if "originalTitle" in df.columns and "primaryTitle" in df.columns:
            same = df.filter(F.col("primaryTitle") == F.col("originalTitle")).count()
            ratio = same / total if total else 0
            print(f"\n  primaryTitle == originalTitle: {ratio * 100:.1f}% збігів")
            if ratio > 0.75:
                cols_to_drop.append("originalTitle")
                print("  → originalTitle вилучається (>75% дублює primaryTitle)")

    elif name == "name_basics":
        for col_name in ["birthYear", "deathYear"]:
            if col_name in df.columns and null_fracs.get(col_name, 0) > null_threshold:
                cols_to_drop.append(col_name)

    cols_to_drop = sorted(set(cols_to_drop))

    if cols_to_drop:
        print(f"\n  Вилучаємо {len(cols_to_drop)} колонок: {cols_to_drop}")
        df = df.drop(*cols_to_drop)
    else:
        print("\n  Неінформативних колонок не знайдено ✓")

    print(f"\n  Колонки після очищення: {df.columns}")
    return df


# ─────────────────────────────────────────────────────────────────────
# КРОК 5а — Аналіз та обробка пропусків
# ─────────────────────────────────────────────────────────────────────
def check_missing(df: DataFrame, name: str = "DataFrame") -> DataFrame:
    _section(f"КРОК 5а — ПРОПУЩЕНІ ЗНАЧЕННЯ: {name}")

    total = df.count()
    print(f"\n  Загальна кількість рядків: {total:,}\n")

    for field in df.schema.fields:
        col_name = field.name
        dtype = field.dataType
        null_count = df.filter(F.col(col_name).isNull()).count()
        pct = 100 * null_count / total if total else 0

        if null_count == 0:
            print(f"  {col_name:<30} — пропусків немає ✓")
            continue

        action = "залишено без змін"

        # Числові
        if _is_numeric_type(dtype):
            if name == "title_basics" and col_name == "startYear":
                median_val = df.stat.approxQuantile(col_name, [0.5], 0.01)[0]
                df = df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isNull(), F.lit(int(median_val))).otherwise(F.col(col_name))
                )
                action = f"заповнено медіаною ({median_val:.1f})"

            elif name == "title_basics" and col_name == "runtimeMinutes":
                # Краще по групах titleType
                if "titleType" in df.columns:
                    medians = (
                        df.groupBy("titleType")
                        .agg(F.expr("percentile_approx(runtimeMinutes, 0.5)").alias("runtime_median"))
                    )

                    df = df.join(medians, on="titleType", how="left")
                    df = df.withColumn(
                        "runtimeMinutes",
                        F.when(F.col("runtimeMinutes").isNull(), F.col("runtime_median"))
                         .otherwise(F.col("runtimeMinutes"))
                    ).drop("runtime_median")
                    action = "заповнено медіаною в межах titleType"
                else:
                    median_val = df.stat.approxQuantile(col_name, [0.5], 0.01)[0]
                    df = df.withColumn(
                        col_name,
                        F.when(F.col(col_name).isNull(), F.lit(int(median_val))).otherwise(F.col(col_name))
                    )
                    action = f"заповнено медіаною ({median_val:.1f})"

            else:
                action = "залишено без змін"

        # Рядкові
        elif isinstance(dtype, StringType):
            if name == "name_basics" and col_name == "primaryName":
                df = df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isNull(), F.lit("Unknown")).otherwise(F.col(col_name))
                )
                action = "заповнено 'Unknown'"
            elif name == "title_principals" and col_name in {"job", "characters"}:
                df = df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isNull(), F.lit("Unknown")).otherwise(F.col(col_name))
                )
                action = "заповнено 'Unknown'"
            else:
                action = "пропущено (рядкова ознака залишена без змін)"

        # Масиви та інше
        elif isinstance(dtype, ArrayType):
            action = "пропущено (масив не заповнюється)"
        else:
            action = "пропущено"

        print(f"  {col_name:<30} {null_count:>8,} null ({pct:5.1f}%)  → {action}")

    return df


# ─────────────────────────────────────────────────────────────────────
# КРОК 5б — Видалення дублікатів
# ─────────────────────────────────────────────────────────────────────
def remove_duplicates(df: DataFrame, name: str = "DataFrame", subset: list | None = None) -> DataFrame:
    _section(f"КРОК 5б — ДУБЛІКАТИ: {name}")

    if subset is None:
        if name == "title_principals":
            subset = ["tconst", "ordering", "nconst"]
        elif "nconst" in df.columns and name == "name_basics":
            subset = ["nconst"]
        elif "tconst" in df.columns:
            subset = ["tconst"]
        else:
            subset = [df.columns[0]]

    total_before = df.count()
    df_dedup = df.dropDuplicates(subset)
    total_after = df_dedup.count()
    removed = total_before - total_after

    print(f"\n  Перевірка за колонками: {subset}")
    print(f"  Рядків до   : {total_before:,}")
    print(f"  Рядків після: {total_after:,}")
    print(f"  Видалено     : {removed:,} дублікатів")

    if removed == 0:
        print("  ✓ Дублікатів не знайдено")

    return df_dedup
