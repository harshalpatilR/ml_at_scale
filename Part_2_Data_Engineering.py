from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os


spark = SparkSession\
    .builder\
    .appName("Airline")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","5")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field")\
    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
    .getOrCreate()    


from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))

schema = StructType(
  [
    StructField("FL_DATE", TimestampType(), True),
    StructField("OP_CARRIER", StringType(), True),
    StructField("OP_CARRIER_FL_NUM", StringType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("CRS_DEP_TIME", StringType(), True),
    StructField("DEP_TIME", StringType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("TAXI_OUT", DoubleType(), True),
    StructField("WHEELS_OFF", StringType(), True),
    StructField("WHEELS_ON", StringType(), True),
    StructField("TAXI_IN", DoubleType(), True),
    StructField("CRS_ARR_TIME", StringType(), True),
    StructField("ARR_TIME", StringType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), True),
    StructField("CANCELLATION_CODE", StringType(), True),
    StructField("DIVERTED", DoubleType(), True),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("AIR_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), True),
    StructField("CARRIER_DELAY", DoubleType(), True),
    StructField("WEATHER_DELAY", DoubleType(), True),
    StructField("NAS_DELAY", DoubleType(), True),
    StructField("SECURITY_DELAY", DoubleType(), True),
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)
  ]
)

flight_df = spark.read.csv(
  path="s3a://ml-field/demo/flight-analysis/data/airlines_csv/*",
  header=True,
  schema=schema
)

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,weekofyear

# pad_time = udf(lambda x: x if len(x) == 4 else "0{}".format(x),StringType())

# df.select("CRS_DEP_TIME").\
#   withColumn('pad_time', pad_time("CRS_DEP_TIME")).show()

# This has been added to help with partitioning.
flight_df = flight_df\
  .withColumn('WEEK',weekofyear('FL_DATE').cast('double'))

smaller_data_set = flight_df.select(
  "WEEK",
  "FL_DATE",
  "OP_CARRIER",
  "OP_CARRIER_FL_NUM",
  "ORIGIN",
  "DEST",
  "CRS_DEP_TIME",
  "CRS_ARR_TIME",
  "CANCELLED",
  "CRS_ELAPSED_TIME",
  "DISTANCE"
)

smaller_data_set.show()

smaller_data_set.write.saveAsTable('default.flight_test_table', format='parquet', mode='overwrite', path='s3a://ml-field/demo/ml_at_scale/')


# This is commented out as it has already been run
#smaller_data_set.write.parquet(
#  path="s3a://ml-field/demo/flight-analysis/data/airline_parquet_partitioned/",
#  partitionBy="WEEK",
#  compression="snappy")


# This will write the able to Hive to be used for other SQL services.
#!cp /home/cdsw/hive-site.xml /etc/hadoop/conf/
#smaller_data_set.write.saveAsTable(
#  'default.flight_test_table', 
#  format='parquet', 
#  mode='overwrite', 
#  path='s3a://ml-field/demo/ml_at_scale/'
#  partitionBy="WEEK")

