"""
Enhanced SageMaker Pipeline for Development Environment
This pipeline demonstrates the core components of a SageMaker Pipeline as a Directed Acyclic Graph (DAG):
- ProcessingStep: Data preprocessing and validation
- TrainingStep: Model training with hyperparameters
- EvaluationStep: Model evaluation and metrics generation
- ConditionStep: Conditional logic based on evaluation results

The pipeline showcases step dependencies, data flow, and conditional execution patterns
that are fundamental to understanding SageMaker Pipeline architecture.
"""

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker import get_execution_role
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_development_pipeline(
    role=None,
    bucket_name="your-sagemaker-bucket",
    region="us-east-1"
):
    """
    Create a comprehensive SageMaker Pipeline for development environment.
    
    This function demonstrates the DAG structure by showing how each step
    depends on outputs from previous steps, creating a clear data flow.
    
    Args:
        role (str): SageMaker execution role ARN
        bucket_name (str): S3 bucket for pipeline artifacts
        region (str): AWS region
        
    Returns:
        Pipeline: Configured SageMaker Pipeline
    """
    
    # Get execution role if not provided
    if role is None:
        try:
            role = get_execution_role()
            logger.info(f"Using execution role: {role}")
        except Exception:
            role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            logger.warning(f"Using default role: {role}")
    
    # =================================================================
    # PIPELINE PARAMETERS - Enable dynamic configuration
    # =================================================================
    logger.info("=== Defining Pipeline Parameters ===")
    
    input_data_s3 = ParameterString(
        name="InputDataS3", 
        default_value=f"s3://{bucket_name}/dev-data/",
        description="S3 path containing raw input data"
    )
    
    training_instance_type = ParameterString(
        name="TrainingInstanceType", 
        default_value="ml.m5.large",
        description="Instance type for training job"
    )
    
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large", 
        description="Instance type for processing jobs"
    )
    
    reg_rate = ParameterFloat(
        name="RegularizationRate", 
        default_value=0.05,
        description="Regularization parameter for logistic regression"
    )
    
    min_accuracy_threshold = ParameterFloat(
        name="MinAccuracyThreshold", 
        default_value=0.75,
        description="Minimum accuracy required for model approval"
    )
    
    test_size = ParameterFloat(
        name="TestSize",
        default_value=0.2,
        description="Proportion of data to use for testing"
    )
    
    # =================================================================
    # STEP 1: DATA PROCESSING - Creates foundation for pipeline DAG
    # =================================================================
    logger.info("=== Defining ProcessingStep ===")
    
    # SKLearn processor for data preprocessing
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="mlops-data-preprocessing"
    )
    
    # ProcessingStep - First node in our DAG
    processing_step = ProcessingStep(
        name="DataPreprocessingStep",
        processor=sklearn_processor,
        code="preprocess.py",  # Script to execute
        inputs=[
            ProcessingInput(
                source=input_data_s3,
                destination="/opt/ml/processing/input/"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket_name}/pipeline-dev/train/"
            ),
            ProcessingOutput(
                output_name="test_data", 
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket_name}/pipeline-dev/test/"
            )
        ],
        arguments=[
            "--test_size", test_size
        ]
    )
    
    # =================================================================
    # STEP 2: MODEL TRAINING - Depends on ProcessingStep outputs
    # =================================================================
    logger.info("=== Defining TrainingStep ===")
    
    # SKLearn estimator for model training
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir="scripts",  # Directory containing training scripts
        role=role,
        instance_type=training_instance_type,
        framework_version="1.0-1",
        py_version="py3",
        hyperparameters={
            "reg_rate": reg_rate
        },
        base_job_name="mlops-model-training"
    )
    
    # TrainingStep - Depends on ProcessingStep (demonstrates DAG dependency)
    training_step = TrainingStep(
        name="ModelTrainingStep",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                # This creates a dependency on the ProcessingStep
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # =================================================================
    # STEP 3: MODEL EVALUATION - Depends on both Processing and Training steps
    # =================================================================
    logger.info("=== Defining EvaluationStep ===")
    
    # Evaluation processor (reusing sklearn_processor)
    evaluation_step = ProcessingStep(
        name="ModelEvaluationStep",
        processor=sklearn_processor,
        code="evaluate.py",
        inputs=[
            # Input 1: Trained model from TrainingStep (creates dependency)
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model"
            ),
            # Input 2: Test data from ProcessingStep (creates dependency)
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test_data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source="/opt/ml/processing/output/metrics",
                destination=f"s3://{bucket_name}/pipeline-dev/evaluation/"
            )
        ],
        # PropertyFile enables reading evaluation metrics for conditional logic
        property_files=[
            PropertyFile(
                name="EvaluationReport",
                output_name="evaluation_metrics", 
                path="evaluation.json"
            )
        ]
    )
    
    # =================================================================
    # STEP 4: CONDITIONAL LOGIC - Quality gate based on evaluation
    # =================================================================
    logger.info("=== Defining ConditionStep ===")
    
    # Condition: Check if accuracy meets minimum threshold
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_step.properties.PropertyFiles.EvaluationReport.JsonGet("accuracy"),
        right=min_accuracy_threshold
    )
    
    # Fail step for when model doesn't meet criteria
    fail_step = FailStep(
        name="ModelFailureStep",
        error_message="Model accuracy below threshold. Pipeline terminated."
    )
    
    # Conditional step - demonstrates pipeline branching logic
    condition_step = ConditionStep(
        name="ModelApprovalCondition",
        conditions=[accuracy_condition],
        if_steps=[],  # Could add deployment steps here
        else_steps=[fail_step]  # Fail pipeline if model doesn't meet criteria
    )
    
    # =================================================================
    # PIPELINE ASSEMBLY - Bringing it all together as a DAG
    # =================================================================
    logger.info("=== Assembling Pipeline DAG ===")
    
    # Create the pipeline with all steps
    pipeline = Pipeline(
        name="MLOpsDevPipeline",
        parameters=[
            input_data_s3,
            training_instance_type, 
            processing_instance_type,
            reg_rate,
            min_accuracy_threshold,
            test_size
        ],
        steps=[
            processing_step,    # Step 1: Data preprocessing
            training_step,      # Step 2: Model training (depends on step 1)
            evaluation_step,    # Step 3: Model evaluation (depends on steps 1 & 2)
            condition_step      # Step 4: Conditional approval (depends on step 3)
        ],
        sagemaker_session=None  # Use default session
    )
    
    return pipeline

def demonstrate_pipeline_properties(pipeline):
    """
    Demonstrate key pipeline properties for educational purposes.
    Shows students how to inspect pipeline structure and dependencies.
    """
    logger.info("=== Pipeline Architecture Analysis ===")
    logger.info(f"Pipeline Name: {pipeline.name}")
    logger.info(f"Number of Steps: {len(pipeline.steps)}")
    logger.info(f"Number of Parameters: {len(pipeline.parameters)}")
    
    logger.info("\n=== Step Dependencies (DAG Structure) ===")
    for step in pipeline.steps:
        logger.info(f"Step: {step.name}")
        logger.info(f"  Type: {type(step).__name__}")
        
        # Show input dependencies for each step
        if hasattr(step, 'inputs') and step.inputs:
            logger.info("  Dependencies:")
            for input_item in step.inputs:
                if hasattr(input_item, 'source'):
                    logger.info(f"    - {input_item.source}")
    
    logger.info("\n=== Parameter Configuration ===")
    for param in pipeline.parameters:
        logger.info(f"  {param.name}: {param.default_value}")

if __name__ == "__main__":
    """
    Main execution block for creating and analyzing the development pipeline.
    This demonstrates the complete pipeline creation process for educational purposes.
    """
    logger.info("=== SageMaker Pipeline Development Example ===")
    logger.info("This script demonstrates core SageMaker Pipeline concepts:")
    logger.info("1. Pipeline as Directed Acyclic Graph (DAG)")
    logger.info("2. Step dependencies and data flow")
    logger.info("3. Conditional execution based on evaluation metrics")
    logger.info("4. Parameter-driven configuration")
    
    try:
        # Create the development pipeline
        dev_pipeline = create_development_pipeline()
        
        # Demonstrate pipeline properties for educational value
        demonstrate_pipeline_properties(dev_pipeline)
        
        # Print pipeline definition (for educational inspection)
        logger.info("\n=== Pipeline JSON Definition ===")
        logger.info("Pipeline definition can be viewed with: pipeline.definition()")
        logger.info("Pipeline can be executed with: pipeline.start()")
        
        logger.info("\n=== Next Steps for Lab ===")
        logger.info("1. Review the pipeline definition JSON")
        logger.info("2. Identify step dependencies in the DAG")
        logger.info("3. Understand how PropertyFiles enable conditional logic")
        logger.info("4. Explore parameter usage for dynamic configuration")
        
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise
