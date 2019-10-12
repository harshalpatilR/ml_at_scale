from pyspark.sql import SparkSession
import os
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



spark = SparkSession\
    .builder\
    .appName("Airline")\
    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
    .getOrCreate()

#    .config("spark.executor.memory","8g")\
#    .config("spark.executor.cores","4")\
#    .config("spark.driver.memory","6g")\
#    .config("spark.executor.instances","5")\
#    .config("spark.dynamicAllocation.enabled","false")\

# TIP
from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))

schema = StructType([StructField("FL_DATE", TimestampType(), True),
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
StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)])


flight_df = spark.read.csv(
  path="s3a://ml-field/demo/flight-analysis/data/airlines_csv/*",header=True,
  schema=schema)

#  
  
### Soo much faster
flight_df.persist()

## Flights Cancelled by Carrier

## Tip put your filter first

cancel_by_carrier = flight_df\
  .filter("CANCELLED == 1")\
  .groupby("OP_CARRIER")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'count_delays')

  
flight_by_carrier = flight_df\
  .groupby("OP_CARRIER")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'count_total')

  
cancel_by_carrier_percent = flight_by_carrier\
  .join(
    cancel_by_carrier, 
    flight_by_carrier.OP_CARRIER == cancel_by_carrier.OP_CARRIER
  )
  
cancel_by_carrier_percent\
  .withColumn(
    "delay_percent",(
      cancel_by_carrier_percent.count_delays/cancel_by_carrier_percent.count_total
    )*100
  )\
  .sort("delay_percent",ascending=False)\
  .toPandas()


# Plot number of flights per year

## TIP
## This is important, you can run spark.sql functions inside R  

flight_by_year = flight_df\
  .withColumn("year",year("FL_DATE"))\
  .groupby("year")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'count_total')
  
cancel_by_year = flight_df\
  .filter("CANCELLED == 1")\
  .withColumn("year_cancel",year("FL_DATE"))\
  .groupby("year_cancel")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'cancel_total')

cancel_by_year_percent = flight_by_year\
  .join(
    cancel_by_year, 
    flight_by_year.year == cancel_by_year.year_cancel
  )  
  
cancel_by_year_percent = cancel_by_year_percent\
  .withColumn(
    "delay_percent",(
      cancel_by_year_percent.cancel_total/cancel_by_year_percent.count_total
    )*100
  )\
  .sort("year",ascending=False)
  
cancel_by_year_percent_pd = cancel_by_year_percent.toPandas()


def plotter():
  sns.set_style("white",{'axes.axisbelow': False})
  plt.bar( 
    cancel_by_year_percent_pd.year, 
    cancel_by_year_percent_pd.delay_percent,
    align='center', 
    alpha=0.5,
    color='#888888',
  )
  plt.grid(color='#FFFFFF', linestyle='-', linewidth=0.5, axis='y')
  plt.title(
    'Percentage Cancelled Flights by Year',
    color='grey'
  )
  plt.xticks(
    cancel_by_year_percent_pd.year,
    color='grey'
  )
  plt.yticks(color='grey')
  sns.despine(left=True,bottom=True)
plotter()



# Plot number of flights per week  

flight_by_week = flight_df\
  .withColumn("week",weekofyear("FL_DATE"))\
  .groupby("week")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'count_total')
  
cancel_by_week = flight_df\
  .filter("CANCELLED == 1")\
  .withColumn("week_cancel",weekofyear("FL_DATE"))\
  .groupby("week_cancel")\
  .count()\
  .sort("count",ascending=False)\
  .withColumnRenamed('count', 'cancel_total')

cancel_by_week_percent = flight_by_week\
  .join(
    cancel_by_week, 
    flight_by_week.week == cancel_by_week.week_cancel
  )  
  
cancel_by_week_percent = cancel_by_week_percent\
  .withColumn(
    "delay_percent",(
      cancel_by_week_percent.cancel_total/cancel_by_week_percent.count_total
    )*100
  )\
  .sort("week",ascending=False)
  
cancel_by_week_percent_pd = cancel_by_week_percent.toPandas()


def plotter():
  sns.set_style("white",{'axes.axisbelow': False})
  plt.bar( 
    cancel_by_week_percent_pd.week, 
    cancel_by_week_percent_pd.delay_percent,
    align='center', 
    alpha=0.5,
    color='#888888',
  )
  plt.grid(color='#FFFFFF', linestyle='-', linewidth=0.5, axis='y')
  plt.title(
    'Percentage Cancelled Flights by Week',
    color='grey'
  )
  plt.xticks(
    color='grey'
  )
  plt.yticks(color='grey')
  sns.despine(left=True,bottom=True)
plotter()

## TIP Aggregation in the select if no groupby

all_routes = flight_df\
  .withColumn("combo_hash", hash("ORIGIN")+hash("DEST"))\
  .withColumn("combo", concat(col("ORIGIN"),col("DEST")))\
  .groupby("combo_hash")\
  .agg(count("combo_hash").alias("count_all"),first("combo").alias("route_alias"))\
  .sort("count_all",ascending=False)

cancelled_routes_all = flight_df\
  .filter("CANCELLED == 1")\
  .withColumn("combo_hash", hash("ORIGIN")+hash("DEST"))\
  .withColumn("combo", concat(col("ORIGIN"),col("DEST")))\
  .groupby("combo_hash")\
  .agg(count("combo_hash").alias("count"),first("combo").alias("route_alias"))\
  .sort("count",ascending=False)  

cancelled_routes_percentage = cancelled_routes_all\
  .join(
    all_routes,
    cancelled_routes_all.combo_hash == all_routes.combo_hash
  )\
  .withColumn(
    "route_alias", 
    concat(
                subrcol("route_alias"),col("DEST")))

cancelled_routes_percentage <-
  cancelled_routes_all %>% 
  inner_join(all_routes,by="combo_hash") %>%
  mutate(
    route = paste(
          substr(first_val.x,0,3), "<>",dest = substr(first_val.x,4,6),sep = ""
        ), 
    cancelled_percent = count/count_all*100) %>% 
  select(route,count_all,count_all,cancelled_percent) %>%
  arrange(desc(cancelled_percent)) 
  
cancelled_routes_percentage %>% as.data.frame
