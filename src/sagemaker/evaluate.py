"""
Evaluation script for SageMaker ProcessingStep.
Evaluates the trained model using comprehensive metrics and generates evaluation report.
This script demonstrates pipeline dependencies: it consumes outputs from both the 
ProcessingStep (test data) and TrainingStep (trained model) in the DAG.
"""
import argparse
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import json
import os
import logging
import numpy as np

# Configure logging for SageMaker Processing Job
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array, optional): Predicted probabilities
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='binary')),
        "recall": float(recall_score(y_true, y_pred, average='binary')),
        "f1_score": float(f1_score(y_true, y_pred, average='binary')),
        "specificity": float(precision_score(y_true, y_pred, pos_label=0, average='binary')),
    }
    
    # Add AUC-ROC if probabilities are available
    if y_pred_proba is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(y_true))
    })
    
    return metrics

def generate_classification_report(y_true, y_pred, output_dir):
    """
    Generate detailed classification report and save to file.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        output_dir (str): Directory to save the report
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    report_path = os.path.join(output_dir, "classification_report.json")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Classification report saved to: {report_path}")
    return report

def evaluate_model(model_path, test_data_path, output_metrics_path):
    """
    Evaluate the trained model using test data and generate comprehensive metrics.
    This function demonstrates how evaluation steps consume outputs from multiple
    pipeline steps, creating dependencies in the DAG.
    
    Args:
        model_path (str): Path to the trained model (from TrainingStep)
        test_data_path (str): Path to test data (from ProcessingStep)
        output_metrics_path (str): Path to save evaluation metrics
    """
    logger.info("=== SageMaker Pipeline EvaluationStep ===")
    logger.info("This step demonstrates pipeline dependencies:")
    logger.info(f"  - Consumes model from TrainingStep: {model_path}")
    logger.info(f"  - Consumes test data from ProcessingStep: {test_data_path}")
    logger.info(f"  - Produces metrics for downstream steps: {output_metrics_path}")
    
    try:
        # Load test dataset
        logger.info("Loading test dataset...")
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Test dataset shape: {test_data.shape}")
        
        # Prepare features and target
        feature_columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                          'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
        
        X_test = test_data[feature_columns]
        y_test = test_data['Diabetic']
        
        logger.info(f"Test features shape: {X_test.shape}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        # Load the trained model
        logger.info(f"Loading trained model from: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model type: {type(model)}")
        
        # Make predictions
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            logger.info("Prediction probabilities calculated")
        
        # Calculate comprehensive metrics
        logger.info("Calculating evaluation metrics...")
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Log key metrics
        logger.info("=== Model Performance Metrics ===")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        
        if 'auc_roc' in metrics:
            logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        logger.info(f"Confusion Matrix: TP={metrics['true_positives']}, "
                   f"TN={metrics['true_negatives']}, FP={metrics['false_positives']}, "
                   f"FN={metrics['false_negatives']}")
        
        # Create output directory and save metrics
        output_dir = os.path.dirname(output_metrics_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main metrics file
        with open(output_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to: {output_metrics_path}")
        
        # Generate detailed classification report
        classification_report = generate_classification_report(y_test, y_pred, output_dir)
        
        # Save prediction results for potential downstream analysis
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_pred_proba if y_pred_proba is not None else [None] * len(y_pred)
        })
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Detailed predictions saved to: {predictions_path}")
        
        # Model performance summary for pipeline decision making
        performance_summary = {
            "model_ready_for_deployment": metrics['accuracy'] > 0.75 and metrics['f1_score'] > 0.70,
            "performance_tier": "high" if metrics['accuracy'] > 0.85 else "medium" if metrics['accuracy'] > 0.75 else "low",
            "recommendation": "proceed_to_deployment" if metrics['accuracy'] > 0.80 else "retrain_required"
        }
        
        summary_path = os.path.join(output_dir, "performance_summary.json")
        with open(summary_path, "w") as f:
            json.dump(performance_summary, f, indent=2)
        
        logger.info("=== Pipeline Decision Summary ===")
        logger.info(f"Model deployment ready: {performance_summary['model_ready_for_deployment']}")
        logger.info(f"Performance tier: {performance_summary['performance_tier']}")
        logger.info(f"Recommendation: {performance_summary['recommendation']}")
        logger.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model for SageMaker Pipeline")
    parser.add_argument("--model_path", type=str, 
                       default="/opt/ml/processing/input/model/model.joblib",
                       help="Path to trained model file (input from TrainingStep)")
    parser.add_argument("--test_data_path", type=str, 
                       default="/opt/ml/processing/input/test_data/test.csv",
                       help="Path to test data (input from ProcessingStep)")
    parser.add_argument("--output_metrics_path", type=str, 
                       default="/opt/ml/processing/output/metrics/evaluation.json",
                       help="Path to save evaluation metrics (output for downstream steps)")
    
    args = parser.parse_args()
    
    logger.info(f"Arguments received: {vars(args)}")
    logger.info("Starting model evaluation process...")
    
    evaluate_model(args.model_path, args.test_data_path, args.output_metrics_path)
