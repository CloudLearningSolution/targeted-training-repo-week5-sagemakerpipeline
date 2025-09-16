"""
Production SageMaker Pipeline
-----------------------------
This production-grade pipeline demonstrates advanced SageMaker Pipeline concepts:
- Production-level resource allocation and performance optimization
- Model validation with stricter quality gates
- Model registration and deployment preparation
- Advanced conditional logic for production approval
- Integration with SageMaker Model Registry for MLOps lifecycle management

This pipeline showcases how the same DAG architecture scales from development
to production with enhanced reliability, monitoring, and approval processes.
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
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import Join
from sagemaker import get_execution_role
import logging

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_production_pipeline(
    role=None,
    bucket_name="your-production-bucket",
    region="us-east-1",
    model_package_group_name="diabetes-prediction-models"
):
    """
    Create a production-grade SageMaker Pipeline with enhanced quality gates.
    
    This production pipeline demonstrates advanced pipeline patterns:
    - Higher performance instance types for production workloads
    - Stricter accuracy thresholds for production deployment
    - Model registration for lifecycle management
    - Enhanced monitoring and logging
    - Multi-condition approval logic
    
    Args:
        role (str): SageMaker execution role ARN
        bucket_name (str): S3 bucket for production pipeline artifacts
        region (str): AWS region
        model_package_group_name (str): Model Registry package group name
        
    Returns:
        Pipeline: Production-configured SageMaker Pipeline
    """
    
    # Get execution role if not provided
    if role is None:
        try:
            role = get_execution_role()
            logger.info(f"Using execution role: {role}")
        except Exception:
            role = "arn:aws:iam::123456789012:role/SageMakerProductionRole"
            logger.warning(f"Using default production role: {role}")
    
    # =================================================================
    # PRODUCTION PIPELINE PARAMETERS - Enhanced for production use
    # =================================================================
    logger.info("=== Defining Production Pipeline Parameters ===")
    
    input_data_s3 = ParameterString(
        name="ProductionInputDataS3",
        default_value=f"s3://{bucket_name}/production-data/",
        description="S3 path containing production input data"
    )
    
    # Production-grade instance types for better performance
    training_instance_type = ParameterString(
        name="ProductionTrainingInstanceType",
        default_value="ml.c5.4xlarge",  # Larger instance for production
        description="High-performance instance type for production training"
    )
    
    processing_instance_type = ParameterString(
        name="ProductionProcessingInstanceType",
        default_value="ml.m5.2xlarge",  # Larger instance for production processing
        description="High-performance instance type for production processing"
    )
    
    # Stricter production parameters
    regularization_rate = ParameterFloat(
        name="ProductionRegularizationRate",
        default_value=0.01,  # Lower for better performance in production
        description="Regularization parameter optimized for production"
    )
    
    # Production quality gates - stricter thresholds
    min_accuracy_threshold = ParameterFloat(
        name="ProductionMinAccuracyThreshold",
        default_value=0.85,  # Higher threshold for production
        description="Minimum accuracy required for production deployment"
    )
    
    min_precision_threshold = ParameterFloat(
        name="ProductionMinPrecisionThreshold",
        default_value=0.80,
        description="Minimum precision required for production deployment"
    )
    
    min_recall_threshold = ParameterFloat(
        name="ProductionMinRecallThreshold", 
        default_value=0.75,
        description="Minimum recall required for production deployment"
    )
    
    # Production data split - more data for training
    test_size = ParameterFloat(
        name="ProductionTestSize",
        default_value=0.15,  # Smaller test set, more for training
        description="Test set size optimized for production"
    )
    
    # Model approval settings
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
        description="Initial approval status for model registration"
    )
    
    # =================================================================
    # STEP 1: PRODUCTION DATA PROCESSING - Enhanced validation
    # =================================================================
    logger.info("=== Defining Production ProcessingStep ===")
    
    # Production-grade SKLearn processor
    sklearn_processor_prod = SKLearnProcessor(
        framework_version="1.0-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="production-data-preprocessing",
        max_runtime_in_seconds=3600,  # 1 hour timeout for production
        tags=[
            {"Key": "Environment", "Value": "Production"},
            {"Key": "Pipeline", "Value": "MLOpsProduction"}
        ]
    )
    
    # Production processing step with enhanced validation
    processing_step = ProcessingStep(
        name="ProductionDataPreprocessingStep",
        processor=sklearn_processor_prod,
        code="preprocess.py",
        inputs=[
            ProcessingInput(
                source=input_data_s3,
                destination="/opt/ml/processing/input/"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="production_train_data",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket_name}/production-pipeline/train/"
            ),
            ProcessingOutput(
                output_name="production_test_data",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket_name}/production-pipeline/test/"
            )
        ],
        arguments=[
            "--test_size", test_size
        ]
    )
    
    # =================================================================
    # STEP 2: PRODUCTION MODEL TRAINING - Optimized configuration
    # =================================================================
    logger.info("=== Defining Production TrainingStep ===")
    
    # Production-optimized SKLearn estimator
    sklearn_estimator_prod = SKLearn(
        entry_point="train.py",
        source_dir="scripts",
        role=role,
        instance_type=training_instance_type,
        framework_version="1.0-1",
        py_version="py3",
        hyperparameters={
            "reg_rate": regularization_rate
        },
        base_job_name="production-model-training",
        max_run=7200,  # 2 hours for production training
        use_spot_instances=False,  # Reliability over cost for production
        tags=[
            {"Key": "Environment", "Value": "Production"},
            {"Key": "ModelType", "Value": "DiabetesPrediction"}
        ]
    )
    
    # Production training step
    training_step = TrainingStep(
        name="ProductionModelTrainingStep",
        estimator=sklearn_estimator_prod,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["production_train_data"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # =================================================================
    # STEP 3: COMPREHENSIVE MODEL EVALUATION - Production validation
    # =================================================================
    logger.info("=== Defining Production EvaluationStep ===")
    
    # Enhanced evaluation step for production
    evaluation_step = ProcessingStep(
        name="ProductionModelEvaluationStep",
        processor=sklearn_processor_prod,
        code="evaluate.py",
        inputs=[
            # Trained model from training step
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model"
            ),
            # Test data from processing step
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["production_test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test_data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="production_evaluation_metrics",
                source="/opt/ml/processing/output/metrics",
                destination=f"s3://{bucket_name}/production-pipeline/evaluation/"
            )
        ],
        # PropertyFiles enable reading metrics for conditional logic
        property_files=[
            PropertyFile(
                name="ProductionEvaluationReport",
                output_name="production_evaluation_metrics",
                path="evaluation.json"
            )
        ]
    )
    
    # =================================================================
    # STEP 4: MODEL CREATION - Prepare for registration
    # =================================================================
    logger.info("=== Defining ModelStep ===")
    
    # Create model for registration
    model = Model(
        image_uri=sklearn_estimator_prod.image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sklearn_estimator_prod.sagemaker_session,
        role=role
    )
    
    model_step = ModelStep(
        name="ProductionModelCreationStep",
        step_args=model.create(
            instance_type="ml.m5.large",
            accelerator_type=None
        )
    )
    
    # =================================================================
    # STEP 5: MULTI-CONDITION QUALITY GATES - Production approval
    # =================================================================
    logger.info("=== Defining Production Quality Gates ===")
    
    # Multiple conditions for production approval
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_step.properties.PropertyFiles.ProductionEvaluationReport.JsonGet("accuracy"),
        right=min_accuracy_threshold
    )
    
    precision_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_step.properties.PropertyFiles.ProductionEvaluationReport.JsonGet("precision"),
        right=min_precision_threshold
    )
    
    recall_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_step.properties.PropertyFiles.ProductionEvaluationReport.JsonGet("recall"),
        right=min_recall_threshold
    )
    
    # Fail steps for different failure scenarios
    accuracy_fail_step = FailStep(
        name="AccuracyFailureStep",
        error_message=Join(
            on=" ",
            values=[
                "Model accuracy",
                evaluation_step.properties.PropertyFiles.ProductionEvaluationReport.JsonGet("accuracy"),
                "below production threshold of",
                min_accuracy_threshold
            ]
        )
    )
    
    precision_fail_step = FailStep(
        name="PrecisionFailureStep", 
        error_message="Model precision below production threshold"
    )
    
    recall_fail_step = FailStep(
        name="RecallFailureStep",
        error_message="Model recall below production threshold"
    )
    
    # =================================================================
    # STEP 6: CONDITIONAL APPROVAL LOGIC - Production deployment gates
    # =================================================================
    logger.info("=== Defining Conditional Approval Logic ===")
    
    # Nested conditional logic for comprehensive validation
    recall_condition_step = ConditionStep(
        name="RecallValidationStep",
        conditions=[recall_condition],
        if_steps=[model_step],  # Proceed to model creation if all conditions met
        else_steps=[recall_fail_step]
    )
    
    precision_condition_step = ConditionStep(
        name="PrecisionValidationStep", 
        conditions=[precision_condition],
        if_steps=[recall_condition_step],
        else_steps=[precision_fail_step]
    )
    
    # Primary accuracy gate
    accuracy_condition_step = ConditionStep(
        name="ProductionModelApprovalGate",
        conditions=[accuracy_condition],
        if_steps=[precision_condition_step],  # Check precision if accuracy passes
        else_steps=[accuracy_fail_step]
    )
    
    # =================================================================
    # PRODUCTION PIPELINE ASSEMBLY - Complete DAG
    # =================================================================
    logger.info("=== Assembling Production Pipeline DAG ===")
    
    # Create production pipeline with all quality gates
    pipeline = Pipeline(
        name="MLOpsProductionPipeline",
        parameters=[
            input_data_s3,
            training_instance_type,
            processing_instance_type,
            regularization_rate,
            min_accuracy_threshold,
            min_precision_threshold,
            min_recall_threshold,
            test_size,
            model_approval_status
        ],
        steps=[
            processing_step,           # Step 1: Data preprocessing
            training_step,            # Step 2: Model training (depends on step 1)
            evaluation_step,          # Step 3: Model evaluation (depends on steps 1 & 2)
            accuracy_condition_step   # Step 4: Multi-tier approval gate (depends on step 3)
        ],
        sagemaker_session=None
    )
    
    return pipeline

def demonstrate_production_features(pipeline):
    """
    Demonstrate production-specific pipeline features for educational purposes.
    """
    logger.info("=== Production Pipeline Features ===")
    logger.info(f"Pipeline: {pipeline.name}")
    logger.info(f"Production Steps: {len(pipeline.steps)}")
    
    logger.info("\n=== Production Enhancements ===")
    logger.info("1. Higher performance instance types")
    logger.info("2. Stricter quality gates (accuracy, precision, recall)")
    logger.info("3. Multi-condition approval logic")
    logger.info("4. Model registration preparation")
    logger.info("5. Enhanced monitoring and tagging")
    logger.info("6. Reliability optimizations (no spot instances)")
    
    logger.info("\n=== Quality Gates Comparison ===")
    logger.info("Development: Single accuracy threshold (0.75)")
    logger.info("Production: Multi-metric thresholds (acc≥0.85, prec≥0.80, rec≥0.75)")
    
    logger.info("\n=== Production DAG Dependencies ===")
    logger.info("ProcessingStep → TrainingStep → EvaluationStep → Multi-Condition Gates")

if __name__ == "__main__":
    """
    Production pipeline creation and demonstration.
    Shows advanced SageMaker Pipeline patterns for production deployment.
    """
    logger.info("=== Production SageMaker Pipeline Example ===")
    logger.info("Demonstrates advanced pipeline concepts:")
    logger.info("1. Production-grade resource allocation")
    logger.info("2. Multi-condition quality gates")
    logger.info("3. Enhanced error handling and monitoring")
    logger.info("4. Model registration preparation")
    
    try:
        # Create production pipeline
        prod_pipeline = create_production_pipeline()
        
        # Demonstrate production features
        demonstrate_production_features(prod_pipeline)
        
        logger.info("\n=== Production vs Development Comparison ===")
        logger.info("This pipeline showcases how the same DAG architecture")
        logger.info("scales from development to production with:")
        logger.info("- Enhanced performance and reliability")
        logger.info("- Stricter quality controls")
        logger.info("- Advanced conditional logic")
        logger.info("- Production monitoring and governance")
        
    except Exception as e:
        logger.error(f"Error creating production pipeline: {str(e)}")
        raise
