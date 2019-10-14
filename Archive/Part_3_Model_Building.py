#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession    .builder    .appName("Airline ML")    .config("spark.executor.memory","16g")    .config("spark.executor.cores","4")    .config("spark.driver.memory","6g")    .config("spark.executor.instances","5")    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field").getOrCreate()

flight_df=spark.read.parquet(
  "s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/",
)


flight_df = flight_df.na.drop() #.limit(100000)


# In[2]:


from IPython.core.display import HTML
import os
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))


# In[3]:


#spark.stop()


# In[4]:


flight_df.printSchema()


# In[5]:


from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,substring,weekofyear,concat,col

convert_time_to_hour = udf(lambda x: x if len(x) == 4 else "0{}".format(x),StringType())

flight_df = flight_df.withColumn('CRS_DEP_HOUR', substring(convert_time_to_hour("CRS_DEP_TIME"),0,2).cast('double'))
flight_df = flight_df.withColumn('WEEK',weekofyear('FL_DATE').cast('double'))
flight_df = flight_df.withColumn("ROUTE", concat(col("ORIGIN"),col("DEST")))


# In[6]:


flight_df.printSchema()


# In[7]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

numeric_cols = ["CRS_ELAPSED_TIME","DISTANCE","WEEK","CRS_DEP_HOUR"]

op_carrier_indexer = StringIndexer(inputCol ='OP_CARRIER', outputCol = 'OP_CARRIER_INDEXED',handleInvalid="keep")
op_carrier_encoder = OneHotEncoder(inputCol ='OP_CARRIER_INDEXED', outputCol='OP_CARRIER_ENCODED')

origin_indexer = StringIndexer(inputCol ='ORIGIN', outputCol = 'ORIGIN_INDEXED',handleInvalid="keep")
origin_encoder = OneHotEncoder(inputCol ='ORIGIN_INDEXED', outputCol='ORIGIN_ENCODED')

dest_indexer = StringIndexer(inputCol ='DEST', outputCol = 'DEST_INDEXED',handleInvalid="keep")
dest_encoder = OneHotEncoder(inputCol ='DEST_INDEXED', outputCol='DEST_ENCODED')

route_indexer = StringIndexer(inputCol ='ROUTE', outputCol = 'ROUTE_INDEXED',handleInvalid="keep")
route_encoder = OneHotEncoder(inputCol ='ROUTE_INDEXED', outputCol='ROUTE_ENCODED')

input_cols=[
    'OP_CARRIER_ENCODED',
    #'ROUTE_INDEXED',
    'ORIGIN_ENCODED',
    'DEST_ENCODED'] + numeric_cols

assembler = VectorAssembler(
    inputCols = input_cols,
    outputCol = 'features')

from pyspark.ml import Pipeline

pipeline_indexed_only = Pipeline(
    stages=[
        op_carrier_indexer,
        op_carrier_encoder,
        origin_indexer,
        origin_encoder,
        dest_indexer,
        dest_encoder,
        #route_indexer,
        assembler]
)


# In[8]:


pipelineModel = pipeline_indexed_only.fit(flight_df)
model_df = pipelineModel.transform(flight_df)
selectedCols = ['CANCELLED', 'features']# + cols
model_df = model_df.select(selectedCols)
model_df.printSchema()
(train, test) = model_df.randomSplit([0.7, 0.3])


# In[9]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'CANCELLED', maxIter=10)

lrModel = lr.fit(train)


# In[10]:


print(lrModel.explainParams())


# In[11]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictionslr = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="CANCELLED",metricName="areaUnderROC")
evaluator.evaluate(predictionslr)


# In[12]:


import matplotlib.pyplot as plt 
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[13]:


input_cols=[
    'OP_CARRIER_INDEXED',
    #'ROUTE_INDEXED',
    'ORIGIN_INDEXED',
    'DEST_INDEXED'] + numeric_cols

pipelineModel = pipeline_indexed_only.fit(flight_df)
model_df = pipelineModel.transform(flight_df)
selectedCols = ['CANCELLED', 'features']# + cols
model_df = model_df.select(selectedCols)
model_df.printSchema()
(train, test) = model_df.randomSplit([0.7, 0.3])


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

rfclassifier = RandomForestClassifier(labelCol = 'CANCELLED', featuresCol = 'features', maxBins=390)
rfmodel = rfclassifier.fit(train)


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictionsrf = rfmodel.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="CANCELLED",metricName="areaUnderROC")
evaluator.evaluate(predictionsrf)


# In[ ]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10,featuresCol = 'features', labelCol = 'CANCELLED')

gbtModel = gbt.fit(train)


# In[ ]:


predictionsgbt = gbtModel.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="CANCELLED",metricName="areaUnderROC")
evaluator.evaluate(predictionsgbt)


# In[ ]:


spark.stop()

