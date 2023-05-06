from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import logging
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg, expr
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def read_and_save():
    spark = SparkSession.builder.appName('data_engineer_work_sample').config("spark.network.timeout", "1200s").getOrCreate()
    
    etf_path = '../psingh02/stock-market-dataset/etfs/'
    stock_path = '../psingh02/stock-market-dataset/stocks/'
    
    etf_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(etf_path).withColumn("Symbol", F.regexp_extract(F.input_file_name(), "/([^/]+)\.csv$",1))
    stock_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(stock_path).withColumn("Symbol", F.regexp_extract(F.input_file_name(), "/([^/]+)\.csv$",1))
    combined_df = etf_df.union(stock_df)
    symbols_df = spark.read.csv("../psingh02/stock-market-dataset/symbols_valid_meta.csv", header = True, inferSchema = True)
    data = combined_df.join(symbols_df, on = "Symbol")

    data_schema = StructType([
            StructField("Symbol", StringType(), True),
            StructField("Security_Name", StringType(), True),
            StructField("Date", StringType(), True),
            StructField("Open", FloatType(), True),
            StructField("High", FloatType(), True),
            StructField("Low", FloatType(), True),
            StructField("Close", FloatType(), True),
            StructField("Adj_Close", FloatType(), True),
            StructField("Volume", FloatType(), True)])
        
    data = data.withColumn("Date", col("Date").cast("string"))
    data = spark.createDataFrame(data.select("Symbol","Security Name","Date","Open","High","Low","Close","Adj Close","Volume") \
                                .rdd, schema= data_schema)
    data.write.mode("overwrite").parquet("pyspark_data_raw.parquet")

def feature_engineering():
    spark = SparkSession.builder.appName('data_engineer_work_sample').master("local[*]").config("spark.network.timeout", "1200s").getOrCreate()
    data = spark.read.parquet("pyspark_data_raw.parquet")

    window_spec = Window.orderBy(F.col('Date').cast('timestamp')).rowsBetween(-29, 0)
    feat_data = data.withColumn("vol_moving_avg", avg(col("Volume")).over(window_spec))

    window_spec = Window.partitionBy("Symbol").orderBy("Date").rowsBetween(-29, 0)
    feat_data = feat_data.withColumn("adj_close_rolling_med", expr("percentile_approx(`Adj_Close`, 0.5)").over(window_spec))

    feat_data.write.mode("overwrite").parquet("feature_engineered_data.parquet")


def sample_by_symbol(group):
    return group.sample(min(len(group), 100))

def train_model():
    logging.basicConfig(filename='training.log', level=logging.INFO)
    ml_data = pd.read_parquet("/Users/psingh02/feature_engineered_data.parquet")
    ml_data_sampled = ml_data.groupby('Symbol').apply(sample_by_symbol)
    ml_data_sampled = ml_data_sampled.reset_index(drop=True)
    # Assume `data` is loaded as a Pandas DataFrame
    ml_data_sampled['Date'] = pd.to_datetime(ml_data_sampled['Date'])
    ml_data_sampled.set_index('Date', inplace=True)
        
    ml_data_sampled.dropna(inplace=True)
        
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'
    X = ml_data_sampled[features]
    y = ml_data_sampled[target]
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    logger = logging.getLogger(__name__)
    # Train the model
    with joblib.parallel_backend('multiprocessing', n_jobs=-1):
        model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f'Mean Absolute Error: {mae}, Mean Squared Error: {mse}')
    joblib.dump(model, 'data_engineer_work_sample_model.joblib') 
    logging.info('Model saved')


default_args = {
    "owner": "Prateek",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "catchup": False,
    "start_date": datetime(2023, 1, 1)   
}

with DAG(dag_id="ETL_process",
         description="Complete DAG for ETL Process",
         schedule=None,
         default_args=default_args
    ) as dag:
        
        extract_task = PythonOperator(
                task_id = "extract_and_save",
                python_callable = read_and_save,
                dag=dag
        )

        feat_engineering_task = PythonOperator(
               task_id = "feature_engineer_dataset",
               python_callable = read_and_save,
               dag = dag
        )

        train_model_task = PythonOperator(
               task_id = "train_ml_model",
               python_callable=train_model,
               dag = dag
        )



extract_task >> feat_engineering_task >> train_model_task