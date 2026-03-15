from pyspark.sql import SparkSession

BASE = "/ABS/PATH/TO/datasets/imdb"

spark = (
    SparkSession.builder
    .appName("imdb-preview")
    .master("local[*]")
    .getOrCreate()
)

def read_tsv_gz(name: str):
    return (
        spark.read
        .option("header", True)
        .option("sep", "\t")
        .option("nullValue", "\\N")
        .csv(f"{BASE}/{name}")
    )

basics = read_tsv_gz("title.basics.tsv.gz")
ratings = read_tsv_gz("title.ratings.tsv.gz")
crew = read_tsv_gz("title.crew.tsv.gz")
names = read_tsv_gz("name.basics.tsv.gz")

print("=== basics ===")
basics.printSchema()
basics.show(5, truncate=False)

print("=== ratings ===")
ratings.printSchema()
ratings.show(5, truncate=False)

spark.stop()
