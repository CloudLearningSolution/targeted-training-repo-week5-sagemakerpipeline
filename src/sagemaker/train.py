"""
Training script for SageMaker TrainingStep.
Trains a logistic regression model and saves the model artifact.
This script is designed to run within a SageMaker TrainingStep as part of a pipeline DAG.

# TODO: Lab 5.1.1 - Component Identification: This script represents a TrainingStep component
# TODO: Lab 5.1.2 - Purpose Recognition: TrainingStep is for model training workloads
# TODO: Lab 5.1.3 - Architecture Understanding: TrainingStep connects ProcessingStep → EvaluationStep
# TODO: Lab 5.1.5 - High-level Comparison: Dedicated training component vs notebook training cells
"""
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# TODO: Lab 5.1.1 - Component Identification: TrainingStep component type
# TODO: Lab 5.1.2 - Purpose Recognition: Core pipeline step type for model training
# TODO: Lab 5.1.5 - High-level Comparison: Component-based logging vs notebook print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(training_data_path, output_model_path, reg_rate):
    """
    Train a logistic regression model for diabetes prediction.
    
    # TODO: Lab 5.1 - Pipeline Data Dependencies and Step Relationships
    # TODO: This function demonstrates how pipeline components connect:
    # TODO: - INPUT: Receives data from upstream ProcessingStep component
    # TODO: - OUTPUT: Produces model artifacts for downstream EvaluationStep component
    # TODO: - RELATIONSHIP: ProcessingStep → TrainingStep → EvaluationStep
    # TODO: - This creates a DIRECTED flow between modular components
    
    Args:
        training_data_path (str): Input from ProcessingStep component
        output_model_path (str): Output for EvaluationStep component  
        reg_rate (float): Training hyperparameter
    """
    # TODO: Lab 5.1 - Component Architecture vs Traditional ML
    # TODO: TrainingStep Component Execution - contrast with traditional approaches:
    # TODO: TRADITIONAL: Training code mixed with data prep in single notebook
    # TODO: PIPELINE: Isolated, reusable training component with clear inputs/outputs
    logger.info("Starting model training process...")
    logger.info("=== TrainingStep Component Execution ===")
    logger.info(f"Component Type: TrainingStep (Core SageMaker Pipeline Step)")
    logger.info(f"Receives input from: ProcessingStep component")
    logger.info(f"Provides output to: EvaluationStep component")
    logger.info(f"Loading training data from: {training_data_path}")
    
    try:
        # TODO: Lab 5.1 - Modular Component Design
        # TODO: Load data from upstream component (demonstrates component separation)
        # TODO: Lab 5.2.6 - TrainingStep Implementation: Load training data from ProcessingStep
        train_data = pd.read_csv(training_data_path)
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info("✓ Successfully received data from upstream ProcessingStep component")
        
        # TODO: Lab 5.2.6 - TrainingStep Implementation: Configure feature columns and target variable
        # Define feature columns for diabetes prediction dataset
        columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                   'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
        
        X_train = train_data[columns]
        y_train = train_data['Diabetic']
        
        logger.info(f"Feature matrix shape: {X_train.shape}")
        logger.info(f"Target variable distribution:\n{y_train.value_counts()}")
        
        # TODO: Lab 5.1 - Component Purpose and Responsibility
        # TODO: TrainingStep focuses solely on model training (single responsibility)
        # TODO: Compare to traditional ML where training mixed with other concerns
        # TODO: Lab 5.2.6 - TrainingStep Implementation: Execute model training with hyperparameters
        logger.info(f"Training logistic regression with regularization rate: {reg_rate}")
        logger.info("=== Core Training Component Logic ===")
        model = LogisticRegression(C=1 / reg_rate, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)
        
        # TODO: Lab 5.2.6 - TrainingStep Implementation: Calculate training metrics for monitoring
        # Calculate training accuracy for monitoring
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # TODO: Lab 5.1 - Component Output Creation
        # TODO: Save model artifacts for downstream component consumption
        # TODO: This creates the connection point to next component in the pipeline
        # TODO: Lab 5.2.6 - TrainingStep Implementation: Save model artifacts as .joblib file
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(model, output_model_path)
        logger.info(f"Model successfully trained and saved at: {output_model_path}")
        logger.info("✓ Created output for downstream EvaluationStep component")
        
        # Log feature importance (coefficients)
        feature_importance = dict(zip(columns, model.coef_[0]))
        logger.info(f"Feature coefficients: {feature_importance}")
        
        # TODO: Lab 5.1 - Component Completion
        # TODO: TrainingStep component completed - pipeline can proceed to next component
        logger.info("=== TrainingStep Component Completed ===")
        logger.info("Pipeline can now execute EvaluationStep component")
        
    except Exception as e:
        # TODO: Lab 5.1 - Component Error Handling
        # TODO: Component failures affect downstream components in the pipeline
        # TODO: Lab 5.2.8 - Error Handling Implementation: Handle TrainingStep component failures
        logger.error(f"TrainingStep component failed: {str(e)}")
        logger.error("❌ Component failure prevents downstream execution")
        raise

if __name__ == "__main__":
    # TODO: Lab 5.2.1 - Step Configuration: Define TrainingStep argument interface
    # TODO: Lab 5.2.2 - Implementation Details: Configure training script parameters
    # TODO: Lab 5.2.5 - TrainingStep Configuration: Parse CLI arguments for training parameters
    parser = argparse.ArgumentParser(description="Train logistic regression model for SageMaker Pipeline")
    parser.add_argument("--training_data_path", type=str, 
                       default="/opt/ml/input/data/train/train.csv",
                       help="Input from upstream ProcessingStep component")
    parser.add_argument("--output_model_path", type=str, 
                       default="/opt/ml/model/model.joblib",
                       help="Output for downstream EvaluationStep component")
    parser.add_argument("--reg_rate", type=float, 
                       default=0.05,
                       help="Hyperparameter for this component")
    
    # TODO: Lab 5.2.5 - TrainingStep Configuration: Parse arguments and align with pipeline expectations
    args = parser.parse_args()
    
    # TODO: Lab 5.1.1 - Component Identification: TrainingStep for model training workloads
    # TODO: Lab 5.1.2 - Purpose Recognition: Handles ML algorithm execution and model creation
    # TODO: Lab 5.1.3 - Architecture Understanding: Connects data processing to model evaluation
    # TODO: Lab 5.1.5 - High-level Comparison: Dedicated component vs notebook training cells
    logger.info("=== Lab 5.1: TrainingStep Component Overview ===")
    logger.info("Component Identification: TrainingStep")
    logger.info("Purpose Recognition: Model training workload execution")
    logger.info("Architecture Understanding: ProcessingStep → TrainingStep → EvaluationStep")
    logger.info("High-level Comparison: Dedicated component vs notebook training cells")
    
    # TODO: Lab 5.2.6 - TrainingStep Implementation: Execute training logic with configured parameters
    train_model(args.training_data_path, args.output_model_path, args.reg_rate)
