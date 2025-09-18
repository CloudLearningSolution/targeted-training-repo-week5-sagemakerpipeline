"""
Training script for SageMaker TrainingStep.
Trains a logistic regression model and saves the model artifact.
This script is designed to run within a SageMaker TrainingStep as part of a pipeline DAG.
"""
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# Configure logging for better observability in SageMaker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(training_data_path, output_model_path, reg_rate):
    """
    Train a logistic regression model for diabetes prediction.
    
    Args:
        training_data_path (str): Path to training CSV file
        output_model_path (str): Path where trained model will be saved
        reg_rate (float): Regularization parameter (inverse of C)
    """
    logger.info("Starting model training process...")
    logger.info(f"Loading training data from: {training_data_path}")
    
    try:
        train_data = pd.read_csv(training_data_path)
        logger.info(f"Training data shape: {train_data.shape}")
        
        # Define feature columns for diabetes prediction dataset
        columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                   'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
        
        X_train = train_data[columns]
        y_train = train_data['Diabetic']
        
        logger.info(f"Feature matrix shape: {X_train.shape}")
        logger.info(f"Target variable distribution:\n{y_train.value_counts()}")
        
        # Train logistic regression model with specified regularization
        logger.info(f"Training logistic regression with regularization rate: {reg_rate}")
        model = LogisticRegression(C=1 / reg_rate, solver="liblinear", random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate training accuracy for monitoring
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Save the trained model
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(model, output_model_path)
        logger.info(f"Model successfully trained and saved at: {output_model_path}")
        
        # Log feature importance (coefficients)
        feature_importance = dict(zip(columns, model.coef_[0]))
        logger.info(f"Feature coefficients: {feature_importance}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression model for SageMaker Pipeline")
    parser.add_argument("--training_data_path", type=str, 
                       default="/opt/ml/input/data/train/train.csv",
                       help="Path to training data CSV file")
    parser.add_argument("--output_model_path", type=str, 
                       default="/opt/ml/model/model.joblib",
                       help="Path to save the trained model")
    parser.add_argument("--reg_rate", type=float, 
                       default=0.05,
                       help="Regularization rate (inverse of C parameter)")
    
    args = parser.parse_args()
    
    logger.info("=== SageMaker Pipeline TrainingStep ===")
    logger.info(f"Arguments received: {vars(args)}")
    
    train_model(args.training_data_path, args.output_model_path, args.reg_rate)
