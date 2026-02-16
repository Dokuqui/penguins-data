from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col


spark = (
    SparkSession.builder.appName("PenguinClassificationFinal")
    .config("spark.cassandra.connection.host", "penguin_cassandra")
    .config("spark.cassandra.connection.port", "9042")
    .getOrCreate()
)

mongo_uri = "mongodb://penguin_mongo:27017/penguin_db.penguins"
mongo_output_uri = "mongodb://penguin_mongo:27017/penguin_db.predictions"

raw_df = spark.read.format("mongodb").option("connection.uri", mongo_uri).load()

df = raw_df.select(
    "penguin_id",
    "label",
    "island",
    col("features.bill_length").alias("bill_length"),
    col("features.bill_depth").alias("bill_depth"),
    col("features.flipper_length").alias("flipper_length"),
    col("features.body_mass").alias("body_mass"),
).dropna()

label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
feature_cols = ["bill_length", "bill_depth", "flipper_length", "body_mass"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features_vec")
rf = RandomForestClassifier(
    labelCol="label_index", featuresCol="features_vec", numTrees=20
)

pipeline = Pipeline(stages=[label_indexer, assembler, scaler, rf])
model = pipeline.fit(df)
predictions = model.transform(df)

predictions.select("penguin_id", "label", "prediction").write.format("mongodb").option(
    "connection.uri", mongo_output_uri
).mode("overwrite").save()

predictions.select(
    "island",
    col("label").alias("species"),
    "penguin_id",
    "bill_length",
    "body_mass",
    "prediction",
).write.format("org.apache.spark.sql.cassandra").options(
    table="penguins_by_island", keyspace="penguin_ks"
).option("spark.cassandra.connection.host", "penguin_cassandra").mode("append").save()
