"""
Preprocessing script for SageMaker ProcessingStep.
Loads raw data, performs data validation and cleaning, splits into train/test, and saves the results.
This script runs as a ProcessingStep in the SageMaker Pipeline DAG, creating data dependencies
for downstream TrainingStep and EvaluationStep components.
"""
import argparse
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

# Configure logging for SageMaker Processing Job
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(df):
    """
    Validate the input dataset and log data quality metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe to validate
        
    Returns:
        pd.DataFrame: Validated and cleaned dataframe
    """
    logger.info("=== Data Validation ===")
    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values}")
    
    # Check data types
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Basic statistics
    logger.info(f"Dataset statistics:\n{df.describe()}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")
    
    # Remove duplicates if any
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
    
    # Validate expected columns for diabetes dataset
    expected_columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                       'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 
                       'Age', 'Diabetic']
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    # Check target variable distribution
    target_dist = df['Diabetic'].value_counts()
    logger.info(f"Target variable distribution:\n{target_dist}")
    logger.info(f"Target variable balance: {target_dist.min() / target_dist.max():.3f}")
    
    return df

def preprocess_data(input_path, output_train, output_test, test_size=0.2):
    """
    Preprocess the diabetes dataset: validate, clean, and split into train/test sets.
    
    Args:
        input_path (str): Path to input data directory
        output_train (str): Path to save training data
        output_test (str): Path to save test data
        test_size (float): Proportion of data to use for testing
    """
    logger.info("=== SageMaker Pipeline ProcessingStep ===")
    logger.info(f"Starting data preprocessing...")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output train path: {output_train}")
    logger.info(f"Output test path: {output_test}")
    
    try:
        # Load the dataset
        input_file = os.path.join(input_path, "diabetes.csv")
        logger.info(f"Loading dataset from: {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate and clean data
        df = validate_data(df)
        
        # Split data using stratified sampling to maintain target distribution
        logger.info(f"Splitting data with test_size={test_size}")
        train_data, test_data = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['Diabetic']  # Maintain target distribution in both sets
        )
        
        logger.info(f"Training set shape: {train_data.shape}")
        logger.info(f"Test set shape: {test_data.shape}")
        
        # Create output directories
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_test), exist_ok=True)
        
        # Save the splits
        train_data.to_csv(output_train, index=False)
        test_data.to_csv(output_test, index=False)
        
        logger.info("=== Processing Summary ===")
        logger.info(f"Successfully saved training data to: {output_train}")
        logger.info(f"Successfully saved test data to: {output_test}")
        logger.info(f"Train set target distribution:\n{train_data['Diabetic'].value_counts()}")
        logger.info(f"Test set target distribution:\n{test_data['Diabetic'].value_counts()}")
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for SageMaker Pipeline")
    parser.add_argument("--input_path", type=str, 
                       default="/opt/ml/processing/input/",
                       help="Path to input data directory")
    parser.add_argument("--output_train", type=str, 
                       default="/opt/ml/processing/output/train/train.csv",
                       help="Path to save training data")
    parser.add_argument("--output_test", type=str, 
                       default="/opt/ml/processing/output/test/test.csv",
                       help="Path to save test data")
    parser.add_argument("--test_size", type=float,
                       default=0.2,
                       help="Proportion of data to use for testing (default: 0.2)")
    
    args = parser.parse_args()
    
    logger.info(f"Arguments received: {vars(args)}")
    
    preprocess_data(args.input_path, args.output_train, args.output_test, args.test_size)
