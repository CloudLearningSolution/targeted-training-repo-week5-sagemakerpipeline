"""
Preprocessing script for SageMaker ProcessingStep.
Loads raw data, performs data validation and cleaning, splits into train/test, and saves the results.
This script runs as a ProcessingStep in the SageMaker Pipeline DAG, creating data dependencies
for downstream TrainingStep and EvaluationStep components.

# TODO: Lab 5.1.1 - Component Identification: This script represents a ProcessingStep component
# TODO: Lab 5.1.2 - Purpose Recognition: ProcessingStep is for data preparation workloads
# TODO: Lab 5.1.3 - Architecture Understanding: ProcessingStep typically starts pipeline execution
# TODO: Lab 5.1.5 - High-level Comparison: Dedicated processing component vs notebook data prep cells
"""
import argparse
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

# TODO: Lab 5.1.1 - Component Identification: ProcessingStep component type
# TODO: Lab 5.1.2 - Purpose Recognition: Core pipeline step type for data processing
# TODO: Lab 5.1.5 - High-level Comparison: Component-based processing vs notebook data prep sections
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(df):
    """
    Validate the input dataset and log data quality metrics.
    
    # TODO: Lab 5.1.2 - Purpose Recognition: ProcessingStep component focuses on data quality and preparation
    # TODO: Lab 5.1.5 - High-level Comparison: Component validation vs mixed notebook data prep
    
    Args:
        df (pd.DataFrame): Input dataframe to validate
        
    Returns:
        pd.DataFrame: Validated and cleaned dataframe
    """
    # TODO: Lab 5.1.2 - Purpose Recognition: Components implement quality gates and validation logic
    # TODO: Lab 5.1.4 - Conceptual Relationships: Failed validation affects downstream component execution
    logger.info("=== ProcessingStep Component: Data Validation ===")
    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {list(df.columns)}")
    logger.info("Performing data quality checks for pipeline...")
    
    # TODO: Lab 5.2.1 - Step Configuration: Configure data quality validation parameters
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
    
    # TODO: Lab 5.2.2 - Implementation Details: Implement data cleaning logic
    # Remove duplicates if any
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
    
    # TODO: Lab 5.2.8 - Error Handling Implementation: Validate data schema for pipeline integrity
    expected_columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                       'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 
                       'Age', 'Diabetic']
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"❌ Schema validation failed - missing columns: {missing_cols}")
        logger.error("This would break downstream pipeline components!")
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    logger.info("✓ Schema validation passed - downstream components will receive correct data")
    
    # Check target variable distribution
    target_dist = df['Diabetic'].value_counts()
    logger.info(f"Target variable distribution:\n{target_dist}")
    logger.info(f"Target variable balance: {target_dist.min() / target_dist.max():.3f}")
    
    logger.info("=== Data validation completed ===")
    return df

def preprocess_data(input_path, output_train, output_test, test_size=0.2):
    """
    Preprocess the diabetes dataset: validate, clean, and split into train/test sets.
    
    # TODO: Lab 5.1.3 - Architecture Understanding: ProcessingStep creates outputs for multiple downstream components
    # TODO: Lab 5.1.4 - Conceptual Relationships: ProcessingStep → TrainingStep AND ProcessingStep → EvaluationStep
    # TODO: Lab 5.1.5 - High-level Comparison: Single component vs notebook cell vs dedicated pipeline component
    
    Args:
        input_path (str): External data source
        output_train (str): Output for TrainingStep component
        output_test (str): Output for EvaluationStep component
        test_size (float): Data splitting parameter
    """
    # TODO: Lab 5.1.3 - Architecture Understanding: ProcessingStep often serves as the first component
    logger.info("=== SageMaker Pipeline ProcessingStep Component ===")
    logger.info("Component Type: ProcessingStep")
    logger.info("Architecture Role: Starting component for ML pipeline")
    logger.info("Component Relationships:")
    logger.info("  • Provides training data → TrainingStep component")
    logger.info("  • Provides test data → EvaluationStep component")
    logger.info(f"Starting data preprocessing...")
    
    # TODO: Step 1.13: DAG Root Node Parameter Configuration
    # TODO: These parameters configure the ROOT NODE of the pipeline DAG
    # TODO: Root nodes typically receive fewer input dependencies (often just data sources)
    logger.info("=== ProcessingStep (ROOT NODE) ===")
    logger.info("This is typically the FIRST NODE to execute in the pipeline DAG")
    logger.info("Key DAG Characteristics of this ROOT NODE:")
    logger.info("  • DIRECTED: Creates outputs that flow to TrainingStep and EvaluationStep")
    logger.info("  • ACYCLIC: No inputs from downstream nodes (prevents cycles)")
    logger.info("  • GRAPH: Connected to multiple downstream nodes")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output train path: {output_train}")
    logger.info(f"Output test path: {output_test}")
    
    try:
        # TODO: Lab 5.2.1 - Step Configuration: Configure ProcessingStep input data source
        # TODO: Lab 5.2.2 - Implementation Details: Load external data for processing
        # TODO: Step 1.7: Root Node Data Loading
        # TODO: Load external data - this is how ROOT NODES typically start DAG execution
        input_file = os.path.join(input_path, "diabetes.csv")
        logger.info(f"Loading dataset from: {input_file}")
        logger.info("📥 ROOT NODE: Loading external data to start DAG execution")
        df = pd.read_csv(input_file)
        
        # TODO: Lab 5.1.2 - Purpose Recognition: Each component has focused responsibility and processing logic
        # TODO: Step 1.8: Execute Data Validation for Entire DAG
        # TODO: Validate data that will affect ALL downstream nodes in the DAG
        logger.info("🔍 ROOT NODE: Validating data quality for entire DAG...")
        df = validate_data(df)
        
        # TODO: Lab 5.2.2 - Implementation Details: Configure data splitting logic
        # TODO: Lab 5.2.1 - Step Configuration: Configure train/test split parameters
        # TODO: Step 1.9: Creating Multiple DAG Dependencies
        # TODO: Split data to create TWO DIRECTED OUTPUTS for different downstream nodes
        # TODO: This demonstrates the GRAPH nature of pipelines (one-to-many relationships)
        logger.info(f"Splitting data with test_size={test_size}")
        logger.info("📊 Creating MULTIPLE DAG DEPENDENCIES:")
        logger.info("   • Train data → will create dependency for TrainingStep")
        logger.info("   • Test data → will create dependency for EvaluationStep")
        
        train_data, test_data = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['Diabetic']  # Maintain target distribution in both sets
        )
        
        logger.info(f"Training set shape: {train_data.shape}")
        logger.info(f"Test set shape: {test_data.shape}")
        
        # TODO: Lab 5.2.2 - Implementation Details: Configure ProcessingStep outputs
        # TODO: Lab 5.2.1 - Step Configuration: Configure output paths and directory structure
        # TODO: Step 1.10: Persisting DAG Dependencies
        # TODO: Save outputs that will become INPUTS for downstream DAG nodes
        # TODO: These saved files create the DATA DEPENDENCIES in the pipeline
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_test), exist_ok=True)
        
        # Save the splits
        train_data.to_csv(output_train, index=False)
        test_data.to_csv(output_test, index=False)
        
        # TODO: Lab 5.1.4 - Conceptual Relationships: ProcessingStep completion enables downstream component execution
        # TODO: Step 1.11: DAG Dependency Creation Complete
        # TODO: Log successful creation of data dependencies for downstream nodes
        logger.info("=== ProcessingStep Component Summary ===")
        logger.info(f"Successfully saved training data to: {output_train}")
        logger.info(f"Successfully saved test data to: {output_test}")
        logger.info("✅ Component outputs created for pipeline architecture:")
        logger.info("   • TrainingStep can now execute (has training data)")
        logger.info("   • EvaluationStep can now execute (has test data)")
        logger.info("✅ DIRECTED OUTPUTS created:")
        logger.info("   • Train data dependency for TrainingStep")
        logger.info("   • Test data dependency for EvaluationStep")
        logger.info(f"Train set target distribution:\n{train_data['Diabetic'].value_counts()}")
        logger.info(f"Test set target distribution:\n{test_data['Diabetic'].value_counts()}")
        logger.info("🎯 ProcessingStep component completed - pipeline can proceed")
        logger.info("🎯 ROOT NODE execution completed - downstream DAG nodes can now execute")
        
    except Exception as e:
        # TODO: Lab 5.2.8 - Error Handling Implementation: Handle ProcessingStep failures
        # TODO: Step 1.12: DAG Failure Propagation  
        # TODO: Handle ROOT NODE failures that would prevent entire DAG execution
        # TODO: Failed root nodes mean NO downstream nodes can execute
        logger.error(f"❌ ProcessingStep component failed: {str(e)}")
        logger.error("This failure prevents downstream components from executing")
        logger.error("❌ ROOT NODE FAILED")
        logger.error("This failure prevents ALL downstream DAG nodes from executing")
        logger.error("The entire pipeline DAG cannot proceed")
        raise

if __name__ == "__main__":
    # TODO: Step 1.13: DAG Root Node Parameter Configuration
    # TODO: These parameters configure the ROOT NODE of the pipeline DAG
    # TODO: Root nodes typically receive fewer input dependencies (often just data sources)
    parser = argparse.ArgumentParser(description="Preprocess data for SageMaker Pipeline")
    parser.add_argument("--input_path", type=str, 
                       default="/opt/ml/processing/input/",
                       help="Path to input data directory (external data source)")
    parser.add_argument("--output_train", type=str, 
                       default="/opt/ml/processing/output/train/train.csv",
                       help="Path to save training data (creates dependency for TrainingStep)")
    parser.add_argument("--output_test", type=str, 
                       default="/opt/ml/processing/output/test/test.csv",
                       help="Path to save test data (creates dependency for EvaluationStep)")
    parser.add_argument("--test_size", type=float,
                       default=0.2,
                       help="Proportion of data to use for testing (affects downstream node inputs)")
    
    args = parser.parse_args()
    
    # TODO: Step 1.14: Pipeline DAG Execution Analysis  
    # TODO: Display how this ROOT NODE fits into the overall DAG structure
    logger.info("=== Pipeline DAG Structure Analysis ===")
    logger.info("This ProcessingStep demonstrates key DAG concepts:")
    logger.info("")  
    logger.info("🌟 DIRECTED ACYCLIC GRAPH (DAG) Characteristics:")
    logger.info("   📍 DIRECTED: Data flows in one direction only")
    logger.info("      • Input: Raw data files")
    logger.info("      • Output 1: Training data → TrainingStep")  
    logger.info("      • Output 2: Test data → EvaluationStep")
    logger.info("")
    logger.info("   🔄 ACYCLIC: No circular dependencies")
    logger.info("      • TrainingStep cannot send data back to ProcessingStep")
    logger.info("      • EvaluationStep cannot send data back to ProcessingStep")
    logger.info("      • This prevents infinite loops in pipeline execution")
    logger.info("")
    logger.info("   📊 GRAPH: Visual representation of step relationships")
    logger.info("      • This node connects to 2 downstream nodes")
    logger.info("      • Creates a branching structure in the pipeline")
    logger.info("      • Demonstrates one-to-many relationships")
    logger.info("")
    logger.info("📋 Data Dependencies Created by This Step:")
    logger.info('   • step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri')
    logger.info('   • step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri')
    logger.info("   These properties become INPUTS for downstream steps")
    logger.info("")
    logger.info(f"Arguments received: {vars(args)}")
    
    # TODO: Step 1.15: Execute Root DAG Node
    # TODO: Begin execution of the ROOT NODE in the pipeline DAG
    logger.info("🚀 Starting ROOT NODE execution...")
    preprocess_data(args.input_path, args.output_train, args.output_test, args.test_size)
