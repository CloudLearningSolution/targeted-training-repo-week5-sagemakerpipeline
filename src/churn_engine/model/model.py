"""
Model training helper for Churn lab exercises (Lab 6.1).

Primary learning objective
--------------------------
Provide an exploration-only reference of a compact model training flow that highlights
where Amazon S3 integration and persistence choices happen. This file is NOT meant
to be edited by learners in this review. Learners should inspect and record WHERE,
WHAT, and WHY for each anchor (no code changes required).

Student expectation
-------------------
For each searchable anchor below record:
- WHERE: exact code line or comment in this file
- WHAT: S3 read/write patterns and joblib usage present in the surrounding lines
- WHY: operational trade-offs (durability, cost, IAM permissions, reproducibility)

High-level exploration anchors (searchable)
-------------------------------------------
# TODO: Lab 6.1.1 - Line-by-Line Import Exploration
# TODO: Lab 6.1.2 - Execution Model Translation
# TODO: Lab 6.1.3 - Parameter and Artifact Evolution
# TODO: Lab 6.1.4 - S3 Data Loading Conversion
# TODO: Lab 6.1.5 - Artifact Persistence and Model Registry Integration
# TODO: Lab 6.1.6 - Logging and Metrics Evolution
"""

import os
import time
import joblib
import logging
import datetime

# TODO: Lab 6.1.1 - Line-by-Line Import Exploration
# WHERE: imports above. WHAT: joblib used locally; boto3 would be used for S3 integration if present.
# WHY: Annotate which imports imply local vs cloud patterns and which packages imply IAM/network concerns.

from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Model:
    """
    Read-only, exploration Model trainer used in lab exercises.

    Purpose
    - Provide a compact example of training + persistence patterns learners should review.
    - Surface where S3, dat-engine, and joblib interactions occur without requiring edits.

    Usage (read-only reference):
        model_trainer = Model(train_filtered)
        model_loaded, final_feature_columns = model_trainer.train(dat=dat, output_file_name_suffix=output_suffix)

    Notes
    - This module intentionally shows both local joblib persistence and an example S3 upload flow
      implemented as a read-only pattern for learners to inspect.
    """

    def __init__(self, train_df: pd.DataFrame, label_col: str = "churn_flag"):
        self.train_df = train_df.copy() if train_df is not None else pd.DataFrame()
        self.label_col = label_col

    def _select_features(self):
        """
        Basic numeric feature selector. Exploration-only.

        TODO: Lab 6.1.3 - Parameter and Artifact Evolution
        WHERE: this function
        WHAT: feature selection is numeric-dtype-based and excludes common id columns
        WHY: keep features stable for reproducibility; annotate how feature evolution affects artifacts
        """
        numeric_cols = self.train_df.select_dtypes(include=["number"]).columns.tolist()
        exclude = {self.label_col, "cust_nbr", "co_nbr", "cust_skey", "co_skey", "yearmo"}
        features = [c for c in numeric_cols if c not in exclude]
        LOG.info("Selected %d feature columns for training", len(features))
        return features

    def _local_persist(self, estimator, model_path: str):
        """
        Persist model locally using joblib (exploration-only).

        WHERE: _local_persist
        WHAT: joblib.dump usage for local artifact persistence
        WHY: fast local IO for dev and reproducibility; fewer IAM concerns but not durable across machines
        """
        joblib.dump(estimator, model_path)
        LOG.info("Model persisted locally to %s", model_path)
        return joblib.load(model_path)

    def _s3_persist(self, estimator, s3_bucket: str, s3_key: str, aws_session=None):
        """
        Example S3 persistence pattern (exploration-only).

        TODO: Lab 6.1.4 - S3 Data Loading Conversion
        WHERE: _s3_persist
        WHAT: temporary local serialization + boto3.upload_file pattern
        WHY: durable artifact storage and cross-region access; requires IAM permissions and has cost/latency tradeoffs

        NOTE: This function exists for learners to inspect S3 upload semantics. In-class runs may not execute it.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except Exception as e:
            LOG.error("boto3 not available in environment: %s", e)
            raise

        tmp_name = f"/tmp/model-{int(time.time())}.joblib"
        joblib.dump(estimator, tmp_name)
        LOG.info("Model serialized to local temp file %s before S3 upload", tmp_name)

        s3 = aws_session.client("s3") if aws_session else boto3.client("s3")
        try:
            s3.upload_file(tmp_name, s3_bucket, s3_key)
            LOG.info("Model uploaded to s3://%s/%s", s3_bucket, s3_key)
        except ClientError as ex:
            LOG.exception("Failed to upload model to S3: %s", ex)
            raise
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

        return f"s3://{s3_bucket}/{s3_key}"

    def train(
        self,
        dat,
        output_file_name_suffix: str = "",
        estimator: Optional[object] = None,
        dry_run: bool = False,
        s3_bucket: Optional[str] = None,
        s3_key_prefix: Optional[str] = None,
        aws_session=None,
    ):
        """
        Train a model and persist artifact. Exploration-only reference; do NOT modify.

        This function demonstrates:
        - how an execution engine may provide dat.get_handle(...) vs manual S3 uploads,
        - where joblib.dump is used locally,
        - an S3 upload branch for durable storage.

        TODO: Lab 6.1.2 - Execution Model Translation
        WHERE: function signature and model persistence branches
        WHAT: dat.get_handle vs direct boto3 uploads; joblib for local serialization
        WHY: help learners reason about local dev vs pipeline engine behaviors
        """
        if self.train_df.empty:
            raise ValueError("No training data available in Model object")

        features = self._select_features()
        if len(features) == 0:
            raise ValueError("No numeric features detected after selection. Inspect train_df columns.")

        if self.label_col not in self.train_df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in training DataFrame")

        X = self.train_df[features]
        y = self.train_df[self.label_col]

        LOG.info("Training shape X: %s y: %s", X.shape, y.shape)

        if dry_run:
            LOG.info("Dry run requested - validating shapes and returning (no training).")
            return None, features

        if estimator is None:
            estimator = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)

        t0 = time.time()
        estimator.fit(X, y)
        elapsed = time.time() - t0
        LOG.info("Model training completed in %.2fs", elapsed)

        model_file_name = f"xgboostmodel_{output_file_name_suffix}.pkl" if output_file_name_suffix else "xgboostmodel.pkl"

        model_path = None
        try:
            model_path = dat.get_handle(model_file_name)
            LOG.info("Using dat.get_handle to persist model at %s", model_path)
        except Exception:
            LOG.warning("dat.get_handle unavailable or failed; falling back to local path for %s", model_file_name)
            model_path = os.path.abspath(model_file_name)

        if model_path.startswith("s3://"):
            # TODO: Lab 6.1.5 - Artifact Persistence and Model Registry Integration
            # WHERE: handling of model_path that contains s3://
            # WHAT: pattern: write locally then upload to S3 (either via dat.upload or boto3)
            # WHY: distinguish engine-managed S3 vs manual upload decisions for permissions and cost
            local_tmp = f"/tmp/{model_file_name}"
            joblib.dump(estimator, local_tmp)
            LOG.info("Wrote model to local tmp %s before copying to engine-managed S3 path %s", local_tmp, model_path)
            try:
                if hasattr(dat, "upload"):
                    dat.upload(local_tmp, model_path)
                    LOG.info("Uploaded model to engine-managed S3 via dat.upload")
                    model_loaded = None
                else:
                    import boto3
                    s3 = aws_session.client("s3") if aws_session else boto3.client("s3")
                    _, _, bucket_and_key = model_path.partition("s3://")
                    bucket, _, key = bucket_and_key.partition("/")
                    s3.upload_file(local_tmp, bucket, key)
                    LOG.info("Uploaded model to %s using boto3", model_path)
                    model_loaded = None
            except Exception as ex:
                LOG.exception("Failed to upload model to engine-managed S3: %s", ex)
                raise
            finally:
                try:
                    os.remove(local_tmp)
                except Exception:
                    pass
            return model_path, features

        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        model_loaded = self._local_persist(estimator, model_path)

        if s3_bucket:
            # NOTE: This branch is for exploration: it shows a manual S3 push after local persist.
            s3_key = f"{s3_key_prefix.rstrip('/')}/{model_file_name}" if s3_key_prefix else model_file_name
            s3_uri = self._s3_persist(estimator, s3_bucket=s3_bucket, s3_key=s3_key, aws_session=aws_session)
            return s3_uri, features

        # TODO: Lab 6.1.6 - Logging and Metrics Evolution
        # WHERE: end of train() before return
        # WHAT: elapsed time, feature count, artifact info are logged
        # WHY: structured logs support reproducibility, cost accounting, and debugging
        LOG.info("Model training finished. Features: %d. Artifact path: %s", len(features), model_path)

        return model_loaded, features
