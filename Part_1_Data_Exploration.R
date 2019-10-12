library(sparklyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(psych)
library(reshape2)
library(leaflet)

## Spark Config

config <- spark_config()
config$spark.hadoop.fs.s3a.aws.credentials.provider  <- "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
config$spark.sql.catalogImplementation <- "in-memory"
sc <- spark_connect(master = "yarn-client", config=config)

s3_link_all <-
  "s3a://ml-field/demo/flight-analysis/data/airlines_csv/*"
#  "s3a://ml-field/demo/flight-analysis/data/airlines_csv/2010.csv"

cols = list(
  FL_DATE = "date",
  OP_CARRIER = "character",
  OP_CARRIER_FL_NUM = "character",
  ORIGIN = "character",
  DEST = "character",
  CRS_DEP_TIME = "character",
  DEP_TIME = "character",
  DEP_DELAY = "double",
  TAXI_OUT = "double",
  WHEELS_OFF = "character",
  WHEELS_ON = "character",
  TAXI_IN = "double",
  CRS_ARR_TIME = "character",
  ARR_TIME = "character",
  ARR_DELAY = "double",
  CANCELLED = "double",
  CANCELLATION_CODE = "character",
  DIVERTED = "double",
  CRS_ELAPSED_TIME = "double",
  ACTUAL_ELAPSED_TIME = "double",
  AIR_TIME = "double",
  DISTANCE = "double",
  CARRIER_DELAY = "double",
  WEATHER_DELAY = "double",
  NAS_DELAY = "double",
  SECURITY_DELAY = "double",
  LATE_AIRCRAFT_DELAY = "double",
  'Unnamed: 27' = "logical"
)

spark_read_csv(
  sc,
  name = "flight_data",
  path = s3_link_all,
  infer_schema = FALSE,
  columns = cols,
  header = TRUE
)

airlines <- tbl(sc, "flight_data")

airlines %>% count()

airlines %>% sample_n(10) %>% as.data.frame

## Flights Cancelled by Carrier

cancelled_flights_by_carrier <-
  airlines %>% 
  group_by(OP_CARRIER) %>% 
  filter(CANCELLED == 1) %>%
  summarise(count_delays = n()) %>%
  arrange(desc(count_delays)) 
  #collect()

flights_by_carrier <-
  airlines %>% 
  group_by(OP_CARRIER) %>% 
  summarise(count = n()) %>%
  arrange(desc(count))
  #collect()

flights_by_carrier %>% 
  left_join(cancelled_flights_by_carrier, by = "OP_CARRIER") %>% 
  mutate(delay_percent = (count_delays/count)*100) %>%
  arrange(desc(delay_percent))



# Plot number of flights per year

## TIP
## This is important, you can run spark.sql functions inside R

flight_counts_by_year <-
  airlines %>% 
  mutate(year = year(FL_DATE)) %>%
  group_by(year) %>% 
  summarise(count = n())

cancel_counts_by_year <-
  airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(year = year(FL_DATE)) %>%
  group_by(year) %>% 
  summarise(count_delays = n())

flight_cancel_percent_year <-
  flight_counts_by_year %>% 
  left_join(cancel_counts_by_year, by = "year") %>% 
  mutate(delay_percent = (count_delays/count)*100) %>%
  arrange(desc(delay_percent))

g <- 
  ggplot(flight_cancel_percent_year, aes(year, delay_percent)) + 
  theme_tufte(base_size=14, ticks=F) + 
  geom_col(width=0.75, fill = "grey") +
  theme(axis.title=element_blank()) +
  scale_x_continuous(breaks=seq(2008,2018,1)) +
  ylab("%") +
  scale_y_continuous() + 
  ggtitle("Percentage Cancelled Flights by Year") + 
  geom_hline(yintercept=seq(1, 2, 1), col="white", lwd=0.5)
plot(g)

## Flights Cancelled by Week of Year

flight_counts_by_week <-
  airlines %>% 
  mutate(week = weekofyear(FL_DATE)) %>%
  group_by(week) %>% 
  summarise(count = n())

cancel_counts_by_week <-
  airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(week = weekofyear(FL_DATE)) %>%
  group_by(week) %>% 
  summarise(count_delays = n())

flight_cancel_percent_week <- 
  flight_counts_by_week %>% 
  left_join(cancel_counts_by_week, by = "week") %>% 
  mutate(delay_percent = (count_delays/count)*100) 

g <- 
  ggplot(flight_cancel_percent_week, aes(week, delay_percent)) + 
  theme_tufte(base_size=14, ticks=F) + 
  geom_col(width=0.75, fill = "grey") +
  theme(axis.title=element_blank()) +
  scale_x_continuous(breaks=seq(1,54,2)) +
  ylab("%") +
  scale_y_continuous() + 
  ggtitle("Percentage Cancelled Flights by Week of Year") + 
  geom_hline(yintercept=seq(1, 4, 1), col="white", lwd=0.5)
plot(g)

## Flights cancelled by Route, for both directions ORIG<>DEST

all_routes <- airlines %>% 
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count_all = n(), first_val = first_value(combo)) %>%
  arrange(desc(count_all)) %>%
  collect

cancelled_routes_all <- airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count = n(), first_val = first_value(combo)) %>%
  arrange(desc(count)) %>%
  collect
 
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

## Cancelled Routes Plotted on a Map

spark_read_csv(
  sc,
  name = "airports",
  path = "s3a://ml-field/demo/flight-analysis/data/airports_orig.csv",
  infer_schema = TRUE,
  header = TRUE
)

airports  <- tbl(sc, "airports")

airports <- airports %>% collect


cancelled_routes_combo <- airlines %>% 
  filter(CANCELLED == 1) %>%
  mutate(combo_hash= hash(ORIGIN) + hash(DEST),combo = paste(ORIGIN,DEST,sep="")) %>% 
  select(combo_hash, combo,ORIGIN, DEST) %>%
  group_by(combo_hash) %>%
  summarize(count = n(), first_val = first_value(combo)) %>%
  arrange(desc(count)) %>%
  collect

cancelled_routes_combo <- cancelled_routes_combo %>% 
  mutate(orig = substr(first_val,0,3), dest = substr(first_val,4,6)) %>%
  select(count,orig,dest)%>%
  inner_join(airports, by=c("orig"="iata"))%>%
  mutate(orig_lat = lat, orig_long = long) %>%
  select(count,orig,dest,orig_lat,orig_long) %>% 
  inner_join(airports, by=c("dest"="iata"))%>%
  mutate(dest_lat = lat, dest_long = long) %>%
  select(count,orig,dest,orig_lat,orig_long,dest_lat,dest_long) %>% 
  filter(between(orig_lat, 20, 50),count > 500)


map3 = leaflet(cancelled_routes_combo) %>% 
  addProviderTiles(providers$CartoDB.Positron)

for(i in 1:nrow(cancelled_routes_combo)){
    map3 <- 
      addPolylines(
        map3, 
        lat = as.numeric(cancelled_routes_combo[i,c(4,6)]), 
        lng = as.numeric(cancelled_routes_combo[i,c(5,7)]),
        weight = 10*(as.numeric(cancelled_routes_combo[i,1]/9881)+0.1), 
        opacity = 0.8*(as.numeric(cancelled_routes_combo[i,1]/9881)+0.05),
        color = "#888"
      )
}
map3

## Which columns are useful?

## TIP

unused_columns <- airlines %>%
  filter(CANCELLED == 1) %>%
  summarise_all(~sum(as.integer(is.na(.)))) %>%
  select_if(~sum(.) > 0) 

unused_columns %>% as.data.frame

unused_columns %>% colnames

#  filter_all(any_vars(. > 1)) %>% as.data.frame


