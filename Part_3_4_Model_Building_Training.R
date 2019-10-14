### Load libraries
library(ggplot2)
library(sparklyr)
library(dplyr)
#library(arrow)

## Connect to Spark. Check spark_defaults.conf for the correct 
##spark_home_set("/etc/spark/")

config <- spark_config()
config$spark.hadoop.fs.s3a.aws.credentials.provider  <- "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
config$spark.executor.memory <- "16g"
config$spark.executor.cores <- "4"
config$spark.driver.memory <- "6g"
config$spark.executor.instances <- "5"
config$spark.sql.catalogImplementation <- "in-memory"
config$spark.yarn.access.hadoopFileSystems <- "s3a://ml-field"
config$spark.dynamicAllocation.enabled  <- "false"


spark <- spark_connect(master = "yarn-client", config=config)

library(cdsw)
html(paste("<a href='http://spark-",Sys.getenv("CDSW_ENGINE_ID"),".",Sys.getenv("CDSW_DOMAIN"),"' target='_blank'>Spark UI<a>",sep=""))

flight_df <- spark_read_parquet(
  spark,
  name = "flight_df",
  path = "s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/",
)


sdf_schema(flight_df) %>% as.data.frame %>% t

flight_df <- flight_df %>% na.omit() 

## TIP UDF
#
#convert_time_to_hour <- function(x) {
#  if (nchar(x) == 4) {
#    return(x)
#    } 
#  else {
#      return(paste("0",x,sep=""))
#    }
#}
#
#flight_df <- flight_df %>% spark_apply(convert_time_to_hour,columns=c("CRS_DEP_HOUR"))


flight_df_mutated <- flight_df %>%
  mutate(CRS_DEP_HOUR = ifelse(length(CRS_DEP_TIME) == 4,CRS_DEP_TIME,paste("0",CRS_DEP_TIME,sep=""))) %>% 
  mutate(CRS_DEP_HOUR = as.numeric(substring(CRS_DEP_HOUR,0,2))) %>%
  mutate(WEEK = as.numeric(weekofyear(FL_DATE))) %>%
  select(CANCELLED,DISTANCE,ORIGIN,DEST,WEEK,CRS_DEP_HOUR,OP_CARRIER,CRS_ELAPSED_TIME)

sdf_schema(flight_df) %>% as.data.frame %>% t

flight_partitions <- flight_df_mutated %>%
  sdf_random_split(training = 0.7, testing = 0.3, seed = 1111)

flights_pipeline <- ml_pipeline(spark) %>%
  ft_string_indexer(
    input_col = "OP_CARRIER",
    output_col = "OP_CARRIER_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "OP_CARRIER_INDEXED",
    output_col = "OP_CARRIER_ENCODED"
  ) %>%
  ft_string_indexer(
    input_col = "ORIGIN",
    output_col = "ORIGIN_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "ORIGIN_INDEXED",
    output_col = "ORIGIN_ENCODED"
  ) %>%
  ft_string_indexer(
    input_col = "DEST",
    output_col = "DEST_INDEXED"
  ) %>%
  ft_one_hot_encoder(
    input_col = "DEST_INDEXED",
    output_col = "DEST_ENCODED"
  ) %>%
  ft_r_formula(
    CANCELLED ~ 
    CRS_ELAPSED_TIME +
    OP_CARRIER_ENCODED +
    CRS_DEP_HOUR + 
    DISTANCE + 
    ORIGIN_ENCODED + 
    DEST_ENCODED + 
    WEEK
  ) %>% 
  ml_logistic_regression(elastic_net_param = 0.0, reg_param = 0.01, max_iter = 15)

fitted_pipeline <- ml_fit(
  flights_pipeline,
  flight_partitions$training
)

predictions <- ml_predict(fitted_pipeline,flight_partitions$testing)

ml_binary_classification_evaluator(predictions)

ml_save(
  fitted_pipeline,
  "s3a://ml-field/demo/flight-analysis/data/models/fitted_pipeline_r",
  overwrite = TRUE
)

