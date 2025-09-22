def evaluate_model(model_path, test_data_path, output_metrics_path):
    """
    Evaluate the trained model using test data and generate comprehensive metrics.
    This function demonstrates how evaluation steps consume outputs from multiple
    pipeline steps, creating dependencies in the DAG.
    
    # TODO: Lab 5.1.3 - Architecture Understanding: EvaluationStep depends on multiple upstream components
    # TODO: Lab 5.1.4 - Conceptual Relationships: Model artifacts (TrainingStep) + Test data (ProcessingStep)
    # TODO: Lab 5.1.5 - High-level Comparison: Dedicated evaluation vs mixed notebook evaluation
    
    Args:
        model_path (str): Input from TrainingStep component
        test_data_path (str): Input from ProcessingStep component
        output_metrics_path (str): Output for ConditionStep component
    """
    # TODO: Lab 5.1.4 - Conceptual Relationships: EvaluationStep demonstrates component convergence pattern
    logger.info("=== SageMaker Pipeline EvaluationStep Component ===")
    logger.info("Component Type: EvaluationStep (ProcessingStep for evaluation)")
    logger.info("Component Dependencies:")
    logger.info(f"📥 Input 1 - Model artifacts from: TrainingStep component")
    logger.info(f"📥 Input 2 - Test data from: ProcessingStep component")
    logger.info("📤 Output - Evaluation metrics for: ConditionStep component")
    logger.info("🔗 Component Relationship Pattern: Many-to-one convergence")
    
    try:
        # TODO: Lab 5.2.2 - Implementation Details: Load and process test data from ProcessingStep
        logger.info("📊 Loading input from ProcessingStep component...")
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Test dataset shape: {test_data.shape}")
        logger.info("✅ Successfully received test data from ProcessingStep component")
        
        # TODO: Lab 5.2.1 - Step Configuration: Configure feature columns for evaluation
        feature_columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                          'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
        
        X_test = test_data[feature_columns]
        y_test = test_data['Diabetic']
        
        logger.info(f"Test features shape: {X_test.shape}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        # TODO: Lab 5.2.2 - Implementation Details: Load and process model from TrainingStep
        logger.info("🤖 Loading input from TrainingStep component...")
        model = joblib.load(model_path)
        logger.info(f"Model type: {type(model)}")
        logger.info("✅ Successfully received trained model from TrainingStep component")
        
        # TODO: Lab 5.2.2 - Implementation Details: Execute model evaluation logic
        logger.info("⚙️ EvaluationStep Component Processing:")
        logger.info("Combining inputs from ProcessingStep (test data) + TrainingStep (model)")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            logger.info("Prediction probabilities calculated")
        
        # TODO: Lab 5.2.2 - Implementation Details: Calculate comprehensive evaluation metrics
        logger.info("📈 Generating output for ConditionStep component...")
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
        
        # TODO: Lab 5.2.1 - Step Configuration: Configure evaluation output paths and formats
        # TODO: Lab 5.2.2 - Implementation Details: Save evaluation metrics for downstream consumption
        output_dir = os.path.dirname(output_metrics_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main metrics file
        with open(output_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📁 Evaluation metrics saved to: {output_metrics_path}")
        logger.info("✅ Created output for ConditionStep component")
        
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
        
        # TODO: Lab 5.2.2 - Implementation Details: Create decision support metrics for ConditionStep
        performance_summary = {
            "model_ready_for_deployment": metrics['accuracy'] > 0.75 and metrics['f1_score'] > 0.70,
            "performance_tier": "high" if metrics['accuracy'] > 0.85 else "medium" if metrics['accuracy'] > 0.75 else "low",
            "recommendation": "proceed_to_deployment" if metrics['accuracy'] > 0.80 else "retrain_required"
        }
        
        summary_path = os.path.join(output_dir, "performance_summary.json")
        with open(summary_path, "w") as f:
            json.dump(performance_summary, f, indent=2)
        
        # TODO: Lab 5.1.4 - Conceptual Relationships: EvaluationStep completion enables ConditionStep execution
        logger.info("=== EvaluationStep Component Summary ===")
        logger.info(f"Model deployment ready: {performance_summary['model_ready_for_deployment']}")
        logger.info(f"Performance tier: {performance_summary['performance_tier']}")
        logger.info(f"Recommendation: {performance_summary['recommendation']}")
        logger.info("✅ EvaluationStep component completed!")
        logger.info("🔄 Pipeline can now execute ConditionStep component")
        
    except Exception as e:
        # TODO: Lab 5.2.8 - Error Handling Implementation: Handle EvaluationStep component failures
        logger.error(f"❌ EvaluationStep component failed: {"""
Evaluation script for SageMaker ProcessingStep.
Evaluates the trained model using comprehensive metrics and generates evaluation report.
This script demonstrates pipeline dependencies: it consumes outputs from both the 
ProcessingStep (test data) and TrainingStep (trained model) in the DAG.

# TODO: Lab 5.1.1 - Component Identification: This script represents an EvaluationStep component
# TODO: Lab 5.1.2 - Purpose Recognition: EvaluationStep (ProcessingStep) is for model evaluation workloads
# TODO: Lab 5.1.3 - Architecture Understanding: EvaluationStep has multiple input dependencies
# TODO: Lab 5.1.5 - High-level Comparison: Dedicated evaluation component vs notebook evaluation cells
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

# TODO: Lab 5.1.1 - Component Identification: EvaluationStep component type
# TODO: Lab 5.1.2 - Purpose Recognition: ProcessingStep used for model evaluation workloads
# TODO: Lab 5.1.5 - High-level Comparison: Component-based evaluation vs notebook evaluation sections
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
    # TODO: Lab 5.2.1 - Step Configuration: Configure EvaluationStep argument interface  
    # TODO: Lab 5.2.2 - Implementation Details: Define EvaluationStep parameter structure
    parser = argparse.ArgumentParser(description="Evaluate trained model for SageMaker Pipeline")
    parser.add_argument("--model_path", type=str, 
                       default="/opt/ml/processing/input/model/model.joblib",
                       help="Input from TrainingStep component")
    parser.add_argument("--test_data_path", type=str, 
                       default="/opt/ml/processing/input/test_data/test.csv",
                       help="Input from ProcessingStep component")
    parser.add_argument("--output_metrics_path", type=str, 
                       default="/opt/ml/processing/output/metrics/evaluation.json",
                       help="Output for ConditionStep component")
    
    args = parser.parse_args()
    
    # TODO: Lab 5.1.1 - Component Identification: EvaluationStep (ProcessingStep for evaluation)
    # TODO: Lab 5.1.2 - Purpose Recognition: Model evaluation and performance assessment
    # TODO: Lab 5.1.3 - Architecture Understanding: Component with multiple input dependencies
    # TODO: Lab 5.1.4 - Conceptual Relationships: Many-to-one component convergence pattern
    # TODO: Lab 5.1.5 - High-level Comparison: Dedicated evaluation vs notebook evaluation sections
    logger.info("=== Lab 5.1: EvaluationStep Component Overview ===")
    logger.info("Component Identification: EvaluationStep (ProcessingStep for evaluation)")
    logger.info("Purpose Recognition: Model evaluation and performance assessment")
    logger.info("Architecture Understanding: Component with multiple input dependencies")
    logger.info("Conceptual Relationships: ProcessingStep → EvaluationStep ← TrainingStep")
    logger.info("High-level Comparison: Dedicated evaluation vs notebook evaluation cells")
    
    # TODO: Lab 5.2.2 - Implementation Details: Execute EvaluationStep with configured parameters
    logger.info("🔄 Executing EvaluationStep component...")
    logger.info("Component waits for ALL upstream dependencies before execution...")
    evaluate_model(args.model_path, args.test_data_path, args.output_metrics_path) concepts:")
    logger.info("")
    logger.info("🔗 CONVERGENCE PATTERN in DAG:")
    logger.info("   Multiple upstream nodes feed into this single node")
    logger.info("   ┌─ ProcessingStep (test data)")
    logger.info("   │")
    logger.info("   ├─→ EvaluationStep ──→ ConditionStep")
    logger.info("   │")
    logger.info("   └─ TrainingStep (model artifacts)")
    logger.info("")
    logger.info("📊 DATA DEPENDENCY Examples:")
    logger.info("   • Model Input Property Reference:")
    logger.info('     training_step.properties.ModelArtifacts.S3ModelArtifacts')
    logger.info("   • Test Data Input Property Reference:")  
    logger.info('     processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri')
    logger.info("")
    logger.info("✅ ACYCLIC VERIFICATION:")
    logger.info("   • EvaluationStep cannot send data back to TrainingStep")
    logger.info("   • EvaluationStep cannot send data back to ProcessingStep") 
    logger.info("   • This maintains DAG acyclic property")
    logger.info("")
    logger.info("📈 OUTPUT DEPENDENCY Creation:")
    logger.info("   • Evaluation metrics will be accessed by ConditionStep via:")
    logger.info('     evaluation_step.properties.PropertyFiles.EvaluationReport.JsonGet("accuracy")')
    logger.info("")
    logger.info(f"Arguments received: {vars(args)}")
    
    # TODO: Step 1.14: Execute Convergence DAG Node
    # TODO: Begin execution of CONVERGENCE NODE that depends on multiple upstream nodes
    logger.info("🔄 Starting CONVERGENCE NODE execution...")
    logger.info("Waiting for ALL upstream DAG dependencies to be satisfied...")
    evaluate_model(args.model_path, args.test_data_path, args.output_metrics_path)
