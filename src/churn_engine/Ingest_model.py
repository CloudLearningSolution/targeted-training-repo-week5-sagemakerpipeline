"""
Ingest-to-model adapter for Churn lab exercises (Lab 6.2).

Primary learning objective
--------------------------
Provide an exploration-only reference showing Redshift -> S3 -> training integration
patterns. This file is NOT meant to be edited by learners. Learners should inspect and
record WHERE, WHAT, and WHY for each anchor (no code changes required).

Student expectation
-------------------
For each searchable anchor below record:
- WHERE: exact code line or comment in this file
- WHAT: Redshift read/write or S3 staging pattern and how training artifact is consumed
- WHY: trade-offs (performance, cost, IAM, reproducibility, scheduling)

High-level exploration anchors (searchable)
-------------------------------------------
# TODO: Lab 6.2.1 - Data Access Pattern Conversion
# TODO: Lab 6.2.2 - ETL Task Mapping
# TODO: Lab 6.2.3 - Orchestration and Dependency Mapping
# TODO: Lab 6.2.4 - Data Movement and Performance Considerations
# TODO: Lab 6.2.5 - Model Training Integration
# TODO: Lab 6.2.6 - Monitoring and Cost Trade-offs
"""

import os
import time
import logging
from typing import Optional, Tuple

import pandas as pd

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def _read_from_redshift(sql_client, sql: str, params: dict = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Exploration helper to read data from Redshift using available SQL access object.

    WHERE: _read_from_redshift
    WHAT: example patterns using sql_client.select_sql_from_dict or pandas.read_sql
    WHY: Redshift is columnar and can be expensive to pull; record trade-offs and auth considerations
    """
    try:
        if hasattr(sql_client, "select_sql_from_dict"):
            q = {"sql": sql, "params": params or {}}
            df = sql_client.select_sql_from_dict(q)
        else:
            df = pd.read_sql(sql, sql_client.conn, params=params)
    except Exception:
        LOG.exception("Redshift read failed; returning empty DataFrame for lab fallback")
        df = pd.DataFrame()
    return df


def stage_table_to_s3(sql_client, table_sql: str, s3_writer, s3_path: str) -> str:
    """
    Exploration-only: show common staging patterns from Redshift -> S3.

    WHERE: stage_table_to_s3
    WHAT: preferred engine-managed UNLOAD vs client-side read -> write patterns
    WHY: UNLOAD is network-efficient; client-side may cause egress and latency/cost implications
    """
    local_tmp = None
    try:
        if hasattr(sql_client, "unload_to_s3"):
            LOG.info("Using engine-managed UNLOAD to export to S3 path %s", s3_path)
            sql_client.unload_to_s3(table_sql, s3_path)
            return s3_path

        df = _read_from_redshift(sql_client, table_sql)
        if df.empty:
            LOG.warning("No rows retrieved to stage to S3")
            return ""
        local_tmp = f"/tmp/stage_{int(time.time())}.parquet"
        df.to_parquet(local_tmp, index=False)
        LOG.info("Wrote local staged file %s", local_tmp)
        if hasattr(s3_writer, "upload"):
            s3_writer.upload(local_tmp, s3_path)
        else:
            try:
                s3_writer.write(local_tmp, s3_path)
            except Exception:
                LOG.warning("s3_writer could not upload; leaving local file at %s", local_tmp)
        return s3_path
    finally:
        try:
            if local_tmp and os.path.exists(local_tmp):
                os.remove(local_tmp)
        except Exception:
            pass


def prepare_redshift_training(
    ingest_obj,
    dat,
    *,
    redshift_sql: str,
    s3_staging_path: Optional[str] = None,
    sample_n: Optional[int] = 100000,
    prefer_unload: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Prepare a training DataFrame by reading from Redshift and optionally staging to S3.

    Exploration-only. Learners should inspect the branching and record WHERE/WHAT/WHY.

    TODO: Lab 6.2.1 - Data Access Pattern Conversion
    WHERE: function signature and first read call
    WHAT: Redshift SQL read -> pandas DataFrame; staging to S3 via UNLOAD or client upload
    WHY: compare direct SQL pulls vs warehouse->S3 staging (cost, speed, auth)
    """
    LOG.info("Preparing training data from Redshift SQL")

    # TODO: Lab 6.2.2 - ETL Task Mapping
    # WHERE: location where ingest_obj methods are referenced (get_labeling_query, etc.)
    # WHAT: map Redshift queries to ETL steps that fill feature tables
    # WHY: show how SQL blocks correspond to ETL tasks in production pipelines

    sql_client = getattr(ingest_obj, "access", None)
    if sql_client is None:
        LOG.warning("Ingest object has no access attribute; attempting to use ingest_obj directly as client")
        sql_client = ingest_obj

    s3_uri = None
    if prefer_unload and s3_staging_path:
        try:
            s3_uri = stage_table_to_s3(sql_client, redshift_sql, dat, s3_staging_path)
            LOG.info("Staged training data to S3 at %s", s3_uri)
            if sample_n and dat:
                try:
                    sample_path = dat.get_handle(s3_uri) if hasattr(dat, "get_handle") else s3_uri
                    sample_df = dat.read(sample_path).head(sample_n) if hasattr(dat, "read") else pd.DataFrame()
                    return sample_df, s3_uri
                except Exception:
                    LOG.warning("Could not read back staged file; returning empty DataFrame and staging uri")
                    return pd.DataFrame(), s3_uri
        except Exception:
            LOG.exception("Engine-managed staging to S3 failed; falling back to direct read")

    df = _read_from_redshift(sql_client, redshift_sql)
    if df.empty:
        LOG.warning("Redshift returned empty DataFrame; check SQL or connectivity")
    else:
        LOG.info("Retrieved %d rows from Redshift", len(df))

    if sample_n and not df.empty and len(df) > sample_n:
        LOG.info("Sampling %d rows for local dev", sample_n)
        df = df.sample(n=sample_n, random_state=42)

    if s3_staging_path and not df.empty:
        try:
            local_tmp = f"/tmp/training_{int(time.time())}.parquet"
            df.to_parquet(local_tmp, index=False)
            if hasattr(dat, "upload"):
                dat.upload(local_tmp, s3_staging_path)
            elif hasattr(dat, "write"):
                dat.write(local_tmp, s3_staging_path)
            else:
                LOG.warning("dat has no upload/write; leaving local file at %s", local_tmp)
            s3_uri = s3_staging_path
            try:
                os.remove(local_tmp)
            except Exception:
                pass
        except Exception:
            LOG.exception("Failed to stage DataFrame to S3")

    # TODO: Lab 6.2.5 - Model Training Integration
    # WHERE: return point of this function
    # WHAT: returning (df, s3_uri) shows two training consumption patterns: direct DF vs staged artifact
    # WHY: compare in-memory training for quick dev vs staged artifacts for reproducible, distributed training

    return df, s3_uri


def orchestration_hint_tasks():
    """
    Exploration-only hints mapping ETL tasks to orchestration steps.

    TODO: Lab 6.2.3 - Orchestration and Dependency Mapping
    WHERE: this function
    WHAT: list of tasks and their dependencies for learners to document
    WHY: show how to express DAGs, retries, idempotency, and scheduling cadence
    """
    return [
        "extract_features: Redshift SELECTs building intermediate feature tables",
        "transform_features: SQL/Glue/Spark transforms to combine feature slices",
        "stage_artifact: Redshift UNLOAD -> S3 or write parquet to S3",
        "train_job: Batch training consuming S3 artifact (distributed) or in-memory DF (dev)",
        "register_model: persist artifact metadata to model registry and versioning system"
    ]


def quick_sample_pipeline(ingest_obj, dat, datadir: str = "/tmp", sample_n: int = 2000):
    """
    Create a small sample training file for classroom iterations (exploration-only).

    TODO: Lab 6.2.4 - Data Movement and Performance Considerations
    WHERE: this helper
    WHAT: creates a small sample locally to avoid full Redshift pulls in classroom
    WHY: reduces cost and execution time for iteration; point out sampling biases
    """
    try:
        label_df = ingest_obj.get_labeling_query()
    except Exception:
        label_df = pd.DataFrame({"cust_nbr": [], "co_nbr": [], "yearmo": [], "churn_flag": []})

    if label_df.empty:
        LOG.warning("Label DF empty; returning empty sample path")
        return pd.DataFrame(), ""

    sample = label_df.head(sample_n).copy()
    sample_path = os.path.join(datadir, f"training_sample_{int(time.time())}.parquet")
    sample.to_parquet(sample_path, index=False)
    LOG.info("Wrote sample training data to %s", sample_path)

    try:
        if hasattr(dat, "upload"):
            remote = f"s3://lab-staging/{os.path.basename(sample_path)}"
            dat.upload(sample_path, remote)
            LOG.info("Uploaded sample to %s", remote)
            return sample, remote
    except Exception:
        LOG.warning("Failed to upload sample to remote storage; returning local path")

    return sample, sample_path


def teaching_checklist():
    """
    Short checklist for learners when reviewing this file (exploration-only).

    TODO: Lab 6.2.6 - Monitoring and Cost Trade-offs
    WHERE: this checklist
    WHAT: metrics, logs, and cost considerations to capture in production
    WHY: teach students to think about observability and economics beyond correctness
    """
    return [
        "Record WHERE Redshift is read in this file (function and line) and how (client call or Unload)",
        "Record WHAT data movement patterns exist (UNLOAD -> S3 vs client read -> upload)",
        "Record WHY each pattern matters for cost, IAM, reproducibility and latency",
        "List orchestration steps required to make this idempotent and testable (retries, checkpoints)",
        "List monitoring signals to emit: rows_extracted, bytes_unloaded, s3_upload_time, sample_count, job_duration",
        "Identify potential cost drivers: Redshift compute, S3 storage tier, cross-region egress, run frequency"
    ]
