"""
src/transformation.py
=====================
Етап трансформації для IMDb dataset.
6 бізнес-питань з filter / join / groupBy / window functions.
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window

SEPARATOR = "=" * 70


def section(title: str):
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


# -------------------------------------------------------------------
# 1. Топ-10 найрейтинговіших фільмів з мінімум 50 000 голосів
# Використано: filter, join
# -------------------------------------------------------------------
def q1_top_rated_movies(df_basics, df_ratings):
    section("БІЗНЕС-ПИТАННЯ 1")
    print("Які 10 найрейтинговіших повнометражних фільмів мають щонайменше 50 000 голосів?")

    result = (
        df_basics
        .filter(F.col("titleType") == "movie")
        .join(df_ratings, on="tconst", how="inner")
        .filter(F.col("numVotes") >= 50000)
        .select("tconst", "primaryTitle", "startYear", "averageRating", "numVotes")
        .orderBy(F.col("averageRating").desc(), F.col("numVotes").desc())
        .limit(10)
    )

    result.show(truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає дві таблиці (Scan).
- Далі застосовує filter до title_basics (лише movie) і до join-результату (numVotes >= 50000).
- Spark Catalyst виносить фільтр numVotes >= 50000 до читання parquet (predicate pushdown),
  завдяки чому зчитується менше даних ще на етапі FileScan.
- Потім виконує SortMergeJoin по tconst — обидві сторони shuffleються по hash(tconst).
- Після join застосовується TakeOrderedAndProject(limit=10) — Spark об'єднує sort і limit
  в одну операцію, уникаючи повного сортування всього датасету.
Основне навантаження: Exchange (shuffle) перед join.
""")

    return result


# -------------------------------------------------------------------
# 2. Які жанри найпоширеніші серед фільмів з рейтингом >= 8.0
# Використано: filter, join, groupBy
# -------------------------------------------------------------------
def q2_top_genres_high_rated(df_basics, df_ratings):
    section("БІЗНЕС-ПИТАННЯ 2")
    print("Які жанри є найпоширенішими серед фільмів з рейтингом не нижче 8.0?")

    result = (
        df_basics
        .filter(F.col("titleType") == "movie")
        .join(df_ratings, on="tconst", how="inner")
        .filter(F.col("averageRating") >= 8.0)
        .withColumn("genre", F.explode("genres"))
        .groupBy("genre")
        .agg(F.count("*").alias("titles_count"))
        .orderBy(F.col("titles_count").desc())
    )

    result.show(20, truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає title_basics (фільтр titleType=movie) та title_ratings (фільтр averageRating>=8.0).
- Оскільки title_ratings після фільтрації невелика, Catalyst обирає BroadcastHashJoin:
  менша таблиця розсилається на всі executor-и, уникаючи shuffle великої таблиці.
- Після join explode(genres) розгортає масив у окремі рядки — це збільшує кількість рядків.
- groupBy(genre) виконується у два етапи (partial_count локально → Exchange → final count),
  що мінімізує обсяг даних при shuffle.
Основне навантаження: explode + Exchange перед фінальною агрегацією.
""")

    return result


# -------------------------------------------------------------------
# 3. Які 10 років мали найбільшу кількість випущених фільмів
# Використано: filter, groupBy
# -------------------------------------------------------------------
def q3_top_years_by_movie_count(df_basics):
    section("БІЗНЕС-ПИТАННЯ 3")
    print("Які 10 років мали найбільшу кількість випущених фільмів?")

    result = (
        df_basics
        .filter(F.col("titleType") == "movie")
        .groupBy("startYear")
        .agg(F.count("*").alias("movies_count"))
        .orderBy(F.col("movies_count").desc())
        .limit(10)
    )

    result.show(truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає лише два стовпці з title_basics (column pruning): titleType та startYear.
- filter(titleType=movie) застосовується ще під час FileScan завдяки predicate pushdown.
- groupBy(startYear) виконується у два етапи: partial_count на кожній партиції локально,
  потім Exchange hashpartitioning(startYear) і фінальний count.
- TakeOrderedAndProject(limit=10) об'єднує sort і limit без повного сортування.
Найпростіший з усіх запитів — один FileScan, один Exchange.

Примітка: 2014 рік показує аномально велику кількість (~127 000 фільмів) порівняно
з іншими роками (~19 000–21 000). Це артефакт даних IMDb, де частина записів
без чіткої дати могла бути масово проставлена одним роком.
""")

    return result


# -------------------------------------------------------------------
# 4. Які фільми входять до топ-3 за рейтингом у межах кожного жанру
# Використано: filter, join, window function
# -------------------------------------------------------------------
def q4_top3_movies_per_genre(df_basics, df_ratings):
    section("БІЗНЕС-ПИТАННЯ 4")
    print("Які фільми входять до топ-3 за рейтингом у межах кожного жанру?")

    window_spec = Window.partitionBy("genre").orderBy(
        F.col("averageRating").desc(),
        F.col("numVotes").desc()
    )

    result = (
        df_basics
        .filter(F.col("titleType") == "movie")
        .join(df_ratings, on="tconst", how="inner")
        .withColumn("genre", F.explode("genres"))
        .withColumn("rank_in_genre", F.row_number().over(window_spec))
        .filter(F.col("rank_in_genre") <= 3)
        .select("genre", "primaryTitle", "startYear", "averageRating", "numVotes", "rank_in_genre")
        .orderBy("genre", "rank_in_genre")
    )

    result.show(100, truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає таблиці, фільтрує movie та виконує SortMergeJoin по tconst.
- explode(genres) збільшує кількість рядків (один рядок → N рядків по жанрах).
- Window row_number() з partitionBy(genre): Spark виконує Exchange hashpartitioning(genre),
  після чого кожна партиція містить лише один жанр і сортується локально.
- Оптимізація WindowGroupLimit: Catalyst додає часткове відсікання топ-3 ще до shuffle
  (Partial WindowGroupLimit), що суттєво зменшує обсяг даних при Exchange.
- Після фінального Window застосовується filter(rank_in_genre <= 3).
Основна складність: два Exchange (join + window) і сортування всередині кожного жанру.
""")

    return result


# -------------------------------------------------------------------
# 5. Які режисери мають найбільшу кількість тайтлів
# Використано: join, groupBy
# -------------------------------------------------------------------
def q5_top_directors_by_titles(df_crew, df_names):
    section("БІЗНЕС-ПИТАННЯ 5")
    print("Які режисери мають найбільшу кількість тайтлів у датасеті?")

    result = (
        df_crew
        .filter(F.col("directors").isNotNull())
        .withColumn("director_id", F.explode("directors"))
        .join(df_names, F.col("director_id") == F.col("nconst"), how="inner")
        .groupBy("director_id", "primaryName")
        .agg(F.count("*").alias("titles_count"))
        .orderBy(F.col("titles_count").desc())
        .limit(20)
    )

    result.show(truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає title_crew (лише колонку directors) та name_basics (nconst, primaryName).
- filter(directors.isNotNull) + перевірка size(directors)>0 застосовується при FileScan.
- explode(directors) перетворює масив у окремі рядки — потенційно велике збільшення даних.
- SortMergeJoin між director_id та nconst: обидві сторони shuffleються по hash ключа.
- groupBy(director_id, primaryName) + count(*) виконується з partial aggregation,
  що зменшує трафік при Exchange.
- TakeOrderedAndProject(limit=20) об'єднує sort і limit.
Найважчі операції: explode (збільшення даних) + два Exchange (join і groupBy).
""")

    return result


# -------------------------------------------------------------------
# 6. Які актори/актриси мають найвищий середній рейтинг серед тих,
#    хто знявся щонайменше у 10 фільмах
# Використано: filter, join, groupBy, window function
#
# ВИПРАВЛЕННЯ: попередня версія використовувала Window без partitionBy,
# що збирало весь датасет на один executor (Exchange SinglePartition).
# Тепер window застосовується після limit — лише для присвоєння rank
# фінальним 20 записам, або замінена на простий orderBy + limit з
# окремим rank через monotonically_increasing_id для відображення.
# -------------------------------------------------------------------
def q6_best_actors_by_avg_rating(df_principals, df_ratings, df_names):
    section("БІЗНЕС-ПИТАННЯ 6")
    print("Які актори/актриси мають найвищий середній рейтинг серед тих, хто знявся щонайменше у 10 фільмах?")

    # Крок 1: агрегація — розраховуємо movies_count та avg_rating для кожного актора
    aggregated = (
        df_principals
        .filter(F.col("category").isin("actor", "actress"))
        .join(df_ratings, on="tconst", how="inner")
        .join(df_names, on="nconst", how="inner")
        .groupBy("nconst", "primaryName")
        .agg(
            F.countDistinct("tconst").alias("movies_count"),
            F.avg("averageRating").alias("avg_rating")
        )
        .filter(F.col("movies_count") >= 10)
    )

    # Крок 2: відбираємо топ-20 через orderBy + limit — без глобального window,
    # що уникає переміщення всіх даних на один executor
    top20 = (
        aggregated
        .orderBy(F.col("avg_rating").desc(), F.col("movies_count").desc())
        .limit(20)
    )

    # Крок 3: додаємо rank лише для фінальних 20 записів — window тут безпечний,
    # бо працює з мінімальним набором даних
    window_spec = Window.orderBy(
        F.col("avg_rating").desc(),
        F.col("movies_count").desc()
    )

    result = top20.withColumn("global_rank", F.row_number().over(window_spec))

    result.show(truncate=False)

    print("\nEXPLAIN:")
    result.explain(True)

    print("""
Аналіз explain():
- Spark читає title_principals (фільтр category in actor/actress), title_ratings, name_basics.
- Два SortMergeJoin: спочатку principals ⋈ ratings по tconst, потім результат ⋈ names по nconst.
  Кожен join потребує Exchange hashpartitioning по ключу join.
- groupBy(nconst, primaryName) з countDistinct(tconst) та avg(averageRating):
  countDistinct вимагає додаткового HashAggregate кроку для дедуплікації tconst,
  що робить цю агрегацію дорожчою за звичайний count.
- filter(movies_count >= 10) відсікає більшість акторів після агрегації.
- TakeOrderedAndProject(limit=20) ефективно відбирає топ-20 без повного сортування.
- Window row_number() застосовується вже до 20 рядків — Exchange SinglePartition
  тут не є проблемою через мінімальний обсяг даних.

Порівняно з попередньою версією: глобальний window на повному агрегованому датасеті
(тисячі акторів) замінено на window лише після limit(20), що усуває попередження
"Moving all data to a single partition" і критичну деградацію продуктивності.
""")

    return result


def run_all_transformations(df_basics, df_ratings, df_names, df_crew, df_principals):
    q1 = q1_top_rated_movies(df_basics, df_ratings)
    q2 = q2_top_genres_high_rated(df_basics, df_ratings)
    q3 = q3_top_years_by_movie_count(df_basics)
    q4 = q4_top3_movies_per_genre(df_basics, df_ratings)
    q5 = q5_top_directors_by_titles(df_crew, df_names)
    q6 = q6_best_actors_by_avg_rating(df_principals, df_ratings, df_names)

    return {
        "q1_top_rated_movies": q1,
        "q2_top_genres_high_rated": q2,
        "q3_top_years_by_movie_count": q3,
        "q4_top3_movies_per_genre": q4,
        "q5_top_directors_by_titles": q5,
        "q6_best_actors_by_avg_rating": q6,
    }
