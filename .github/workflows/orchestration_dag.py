"""
Apache Airflow DAG
Machine Learning Operations Playbook Adoption Workshop – Phase 2:
Data Services Integration Architecture - Hands-On Workshop

This DAG demonstrates orchestration of tasks using model.py (Lab 6.1)
and ingest_model.py (Lab 6.2). It is illustrative only: external
dependencies like dat, ingest_obj, and Redshift connections are mocked.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your lab modules
from churn_engine.model import model as model_module
from churn_engine.model import ingest_model as ingest_module

# -------------------------------------------------------------------
# Placeholder objects to simulate external dependencies
# -------------------------------------------------------------------
class DummyDat:
    def get_handle(self, name):
        return f"/tmp/{name}"
    def upload(self, local, remote):
        print(f"Uploading {local} to {remote}")
    def read(self, path):
        print(f"Reading {path}")
        return []

class DummyIngest:
    def get_labeling_query(self):
        import pandas as pd
        return pd.DataFrame({
            "cust_nbr": ["C1", "C2"],
            "co_nbr": [5, 6],
            "yearmo": [202001, 202002],
            "churn_flag": [0, 1]
        })

dat = DummyDat()
ingest_obj = DummyIngest()

# -------------------------------------------------------------------
# Task functions
# -------------------------------------------------------------------
def run_ingest(**context):
    df, s3_uri = ingest_module.prepare_redshift_training(
        ingest_obj,
        dat,
        redshift_sql="SELECT * FROM churn_training_data",
        s3_staging_path="s3://dummy-bucket/training-data",
        sample_n=100
    )
    print("Ingest complete. Rows:", len(df), "S3 URI:", s3_uri)

def run_model_train(**context):
    import pandas as pd
    # Dummy training data
    df = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1, 0, 1],
        "churn_flag": [0, 1, 0]
    })
    trainer = model_module.Model(df)
    model_loaded, features = trainer.train(dat=dat, output_file_name_suffix="airflow-demo", dry_run=True)
    print("Model training complete. Features:", features)

# -------------------------------------------------------------------
# DAG definition
# -------------------------------------------------------------------
with DAG(
    dag_id="mlops_phase2_data_services",
    description="Phase 2 Data Services Integration Architecture - Hands-On Workshop",
    start_date=datetime(2025, 11, 20),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "phase2", "lab6.1", "lab6.2"]
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_redshift_data",
        python_callable=run_ingest,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id="train_model_s3",
        python_callable=run_model_train,
        provide_context=True
    )

    ingest_task >> train_task
