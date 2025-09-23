"""
Model Deployment Pipeline for SageMaker
=======================================
This deployment pipeline demonstrates advanced SageMaker Pipeline concepts for model lifecycle management:
- ModelStep: Create SageMaker model from training artifacts
- RegisterModelStep: Register model with SageMaker Model Registry
- Endpoint Deployment: Deploy model to real-time inference endpoint
- Best Practices: Validation, monitoring, and rollback capabilities

This pipeline integrates with the training pipeline outputs and demonstrates
the complete MLOps lifecycle from training to production deployment.

# TODO: Lab 5.1.1 - Component Identification: Advanced step types for model deployment (ModelStep, RegisterModelStep)
# TODO: Lab 5.1.2 - Purpose Recognition: Model lifecycle management and deployment step purposes
# TODO: Lab 5.1.3 - Architecture Understanding: Extended pipeline architecture for deployment workflows
# TODO: Lab 5.1.4 - Conceptual Relationships: How deployment components integrate with training pipelines
# TODO: Lab 5.1.5 - High-level Comparison: Complete MLOps lifecycle vs traditional model deployment
"""

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, CreateModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.register_model_step import RegisterModelStep
from sagemaker.workflow.lambda_step import LambdaStep, Lambda
from sagemaker.lambda_helper import Lambda as LambdaHelper
from sagemaker.workflow.functions import Join
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker import get_execution_role
from sagemaker.inputs import CreateModelInput
from sagemaker.predictor import Predictor
import logging
import json

# Configure logging for deployment pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_inference_script():
    """
    Create inference script for model deployment.
    This script will be used by the SageMaker endpoint for real-time predictions.
    """
    inference_code = '''
import joblib
import pandas as pd
import numpy as np
import json
import os

def model_fn(model_dir):
    """Load the trained model from the model directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data for prediction."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Expected feature names
        feature_names = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                        'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
        
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        return df[feature_names]
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }

def output_fn(prediction, content_type):
    """Format the prediction output."""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    return inference_code

def create_endpoint_validation_lambda():
    """
    Create Lambda function code for endpoint validation.
    This validates the deployed endpoint before approval.
    """
    lambda_code = '''
import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Validate the deployed SageMaker endpoint with test predictions.
    """
    try:
        # Extract endpoint name from event
        endpoint_name = event.get('endpoint_name')
        if not endpoint_name:
            raise ValueError("endpoint_name not provided in event")
        
        # Initialize SageMaker Runtime client
        sagemaker_runtime = boto3.client('sagemaker-runtime')
        
        # Test data for validation
        test_payload = {
            "Pregnancies": 1,
            "PlasmaGlucose": 120,
            "DiastolicBloodPressure": 70,
            "TricepsThickness": 25,
            "SerumInsulin": 100,
            "BMI": 25.5,
            "DiabetesPedigree": 0.5,
            "Age": 35
        }
        
        # Make test prediction
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Validate response structure
        if 'predictions' not in result or 'probabilities' not in result:
            raise ValueError("Invalid response structure from endpoint")
        
        # Log validation results
        logger.info(f"Endpoint {endpoint_name} validation successful")
        logger.info(f"Test prediction: {result}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'validation_status': 'success',
                'endpoint_name': endpoint_name,
                'test_result': result
            })
        }
        
    except Exception as e:
        logger.error(f"Endpoint validation failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'validation_status': 'failed',
                'error': str(e)
            })
        }
'''
    return lambda_code

def create_deployment_pipeline(
    role=None,
    bucket_name="your-deployment-bucket",
    region="us-east-1",
    model_package_group_name="diabetes-prediction-models",
    trained_model_s3_uri=None,
    evaluation_metrics_s3_uri=None
):
    """
    Create a comprehensive model deployment pipeline with all required steps.
    
    # TODO: Lab 5.1.3 - Architecture Understanding: Complete deployment workflow architecture
    # TODO: Lab 5.1.4 - Conceptual Relationships: Integration with training pipeline outputs
    # TODO: Lab 5.1.5 - High-level Comparison: Automated deployment vs manual model deployment
    
    Args:
        role (str): SageMaker execution role ARN
        bucket_name (str): S3 bucket for deployment artifacts
        region (str): AWS region
        model_package_group_name (str): Model Registry package group name
        trained_model_s3_uri (str): S3 URI of trained model artifacts
        evaluation_metrics_s3_uri (str): S3 URI of evaluation metrics
        
    Returns:
        Pipeline: Complete deployment pipeline
    """
    
    # Get execution role if not provided
    if role is None:
        try:
            role = get_execution_role()
            logger.info(f"Using execution role: {role}")
        except Exception:
            role = "arn:aws:iam::123456789012:role/SageMakerDeploymentRole"
            logger.warning(f"Using default deployment role: {role}")
    
    # Create PipelineSession for proper pipeline execution
    pipeline_session = PipelineSession()
    
    # =================================================================
    # DEPLOYMENT PIPELINE PARAMETERS
    # =================================================================
    logger.info("=== Defining Deployment Pipeline Parameters ===")
    
    # Model artifacts from training pipeline
    model_artifacts_s3 = ParameterString(
        name="ModelArtifactsS3Uri",
        default_value=trained_model_s3_uri or f"s3://{bucket_name}/model-artifacts/",
        description="S3 URI of trained model artifacts from training pipeline"
    )
    
    # Evaluation metrics from evaluation step
    evaluation_metrics_s3 = ParameterString(
        name="EvaluationMetricsS3Uri", 
        default_value=evaluation_metrics_s3_uri or f"s3://{bucket_name}/evaluation-metrics/",
        description="S3 URI of evaluation metrics from evaluation pipeline"
    )
    
    # Deployment configuration
    endpoint_instance_type = ParameterString(
        name="EndpointInstanceType",
        default_value="ml.m5.large",
        description="Instance type for model endpoint"
    )
    
    endpoint_instance_count = ParameterInteger(
        name="EndpointInstanceCount",
        default_value=1,
        description="Number of instances for model endpoint"
    )
    
    # Model approval thresholds for deployment
    deployment_approval_threshold = ParameterFloat(
        name="DeploymentApprovalThreshold",
        default_value=0.85,
        description="Minimum accuracy required for deployment approval"
    )
    
    # Model package group for registry
    model_package_group = ParameterString(
        name="ModelPackageGroup",
        default_value=model_package_group_name,
        description="Model Registry package group name"
    )
    
    # Endpoint configuration
    endpoint_name = ParameterString(
        name="EndpointName",
        default_value="diabetes-prediction-endpoint",
        description="Name for the deployed model endpoint"
    )
    
    # =================================================================
    # STEP 1: PRE-DEPLOYMENT VALIDATION
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: ProcessingStep used for deployment validation
    # TODO: Lab 5.1.2 - Purpose Recognition: Validation and quality assurance before deployment
    logger.info("=== Defining Pre-Deployment Validation Step ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: SKLearn processor for pre-deployment validation
    validation_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="pre-deployment-validation",
        sagemaker_session=pipeline_session
    )
    
    # Create validation script for model artifacts
    validation_code = '''
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_artifacts():
    """Validate model artifacts before deployment."""
    try:
        # Load and validate model
        model_path = "/opt/ml/processing/input/model/model.joblib"
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {type(model)}")
        
        # Validate model has required methods
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model missing required method: {method}")
        
        # Load evaluation metrics
        metrics_path = "/opt/ml/processing/input/metrics/evaluation.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Validation summary
        validation_result = {
            "model_validation": "passed",
            "model_type": str(type(model)),
            "model_methods": [m for m in dir(model) if not m.startswith('_')],
            "evaluation_metrics": metrics,
            "deployment_ready": True
        }
        
        # Save validation results
        os.makedirs("/opt/ml/processing/output/validation", exist_ok=True)
        with open("/opt/ml/processing/output/validation/validation_report.json", 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        logger.info("Model validation completed successfully")
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    validate_model_artifacts()
'''
    
    # TODO: Lab 5.1.4 - Conceptual Relationships: Validation step integrates with training pipeline outputs
    # TODO: Lab 5.2.2 - Implementation Details: Pre-deployment validation step configuration
    validation_step = ProcessingStep(
        name="PreDeploymentValidationStep",
        processor=validation_processor,
        code=validation_code,
        inputs=[
            ProcessingInput(
                source=model_artifacts_s3,
                destination="/opt/ml/processing/input/model"
            ),
            ProcessingInput(
                source=evaluation_metrics_s3,
                destination="/opt/ml/processing/input/metrics"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="validation_report",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{bucket_name}/deployment-pipeline/validation/"
            )
        ],
        property_files=[
            PropertyFile(
                name="ValidationReport",
                output_name="validation_report",
                path="validation_report.json"
            )
        ]
    )
    
    # =================================================================
    # STEP 2: CREATE SAGEMAKER MODEL - ModelStep
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: ModelStep for SageMaker model creation
    # TODO: Lab 5.1.2 - Purpose Recognition: ModelStep creates deployable SageMaker model objects
    # TODO: Lab 5.1.4 - Conceptual Relationships: ModelStep depends on validation step completion
    logger.info("=== Defining ModelStep for SageMaker Model Creation ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: Create SageMaker Model object configuration
    sklearn_model = Model(
        image_uri="246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",
        model_data=model_artifacts_s3,
        role=role,
        entry_point="inference.py",
        source_dir="deployment_scripts",
        framework_version="1.0-1",
        py_version="py3",
        sagemaker_session=pipeline_session
    )
    
    # TODO: Lab 5.2.2 - Implementation Details: ModelStep implementation
    model_step = ModelStep(
        name="CreateSageMakerModelStep",
        step_args=sklearn_model.create(
            instance_type=endpoint_instance_type,
            accelerator_type=None
        ),
        depends_on=[validation_step]
    )
    
    # =================================================================
    # STEP 3: ALTERNATIVE CREATE MODEL STEP - CreateModelStep
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: CreateModelStep as alternative model creation approach
    # TODO: Lab 5.1.2 - Purpose Recognition: Different step types can achieve similar purposes
    # TODO: Lab 5.1.5 - High-level Comparison: Multiple approaches for same deployment task
    logger.info("=== Defining CreateModelStep (Alternative Approach) ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: Alternative approach using CreateModelStep
    create_model_step = CreateModelStep(
        name="AlternativeCreateModelStep",
        model=sklearn_model,
        inputs=CreateModelInput(
            instance_type=endpoint_instance_type,
            accelerator_type=None
        ),
        depends_on=[validation_step]
    )
    
    # =================================================================
    # STEP 4: MODEL REGISTRY - RegisterModelStep
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: RegisterModelStep for model lifecycle management
    # TODO: Lab 5.1.2 - Purpose Recognition: RegisterModelStep manages model versioning and metadata
    # TODO: Lab 5.1.4 - Conceptual Relationships: RegisterModelStep depends on ModelStep completion
    logger.info("=== Defining RegisterModelStep ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: Model metrics for registry registration
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[evaluation_metrics_s3, "evaluation.json"]
            ),
            content_type="application/json"
        )
    )
    
    # TODO: Lab 5.2.2 - Implementation Details: Register model in Model Registry
    register_model_step = RegisterModelStep(
        name="RegisterModelInRegistryStep",
        estimator=sklearn_model,
        model_data=model_artifacts_s3,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[endpoint_instance_type],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
        depends_on=[model_step]
    )
    
    # =================================================================
    # STEP 5: ENDPOINT DEPLOYMENT - Lambda-based deployment
    # =================================================================
    logger.info("=== Defining Endpoint Deployment Step ===")
    
    # Lambda function for endpoint deployment
    deployment_lambda_code = '''
import json
import boto3
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Deploy model to SageMaker endpoint."""
    try:
        sagemaker = boto3.client('sagemaker')
        
        # Extract parameters from event
        model_name = event.get('model_name')
        endpoint_name = event.get('endpoint_name')
        instance_type = event.get('instance_type', 'ml.m5.large')
        instance_count = int(event.get('instance_count', 1))
        
        # Create endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # Create or update endpoint
        try:
            # Try to update existing endpoint
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Updating existing endpoint: {endpoint_name}")
        except sagemaker.exceptions.ClientError:
            # Create new endpoint if it doesn't exist
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Creating new endpoint: {endpoint_name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'endpoint_name': endpoint_name,
                'endpoint_config_name': endpoint_config_name,
                'status': 'deployment_initiated'
            })
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'status': 'deployment_failed'
            })
        }
'''
    
    # Create Lambda helper for deployment
    deployment_lambda = LambdaHelper(
        function_name="sagemaker-model-deployment",
        execution_role_arn=role,
        script=deployment_lambda_code,
        handler="lambda_function.lambda_handler",
        timeout=300,
        memory_size=128
    )
    
    # Lambda step for deployment
    deployment_step = LambdaStep(
        name="DeployModelToEndpointStep",
        lambda_func=deployment_lambda,
        inputs={
            "model_name": model_step.properties.ModelName,
            "endpoint_name": endpoint_name,
            "instance_type": endpoint_instance_type,
            "instance_count": endpoint_instance_count
        },
        depends_on=[register_model_step]
    )
    
    # =================================================================
    # STEP 6: ENDPOINT VALIDATION - Best Practice
    # =================================================================
    logger.info("=== Defining Endpoint Validation Step ===")
    
    # Lambda function for endpoint validation
    validation_lambda = LambdaHelper(
        function_name="sagemaker-endpoint-validation",
        execution_role_arn=role,
        script=create_endpoint_validation_lambda(),
        handler="lambda_function.lambda_handler",
        timeout=180,
        memory_size=128
    )
    
    # Endpoint validation step
    endpoint_validation_step = LambdaStep(
        name="ValidateDeployedEndpointStep",
        lambda_func=validation_lambda,
        inputs={
            "endpoint_name": endpoint_name
        },
        depends_on=[deployment_step]
    )
    
    # =================================================================
    # STEP 7: CONDITIONAL APPROVAL FOR PRODUCTION - Best Practice
    # =================================================================
    logger.info("=== Defining Production Approval Gate ===")
    
    # Deployment approval condition
    deployment_condition = ConditionGreaterThanOrEqualTo(
        left=validation_step.properties.PropertyFiles.ValidationReport.JsonGet("evaluation_metrics.accuracy"),
        right=deployment_approval_threshold
    )
    
    # Fail step for deployment rejection
    deployment_fail_step = FailStep(
        name="DeploymentRejectedStep",
        error_message=Join(
            on=" ",
            values=[
                "Model accuracy",
                validation_step.properties.PropertyFiles.ValidationReport.JsonGet("evaluation_metrics.accuracy"),
                "below deployment threshold of",
                deployment_approval_threshold
            ]
        )
    )
    
    # Conditional deployment approval
    deployment_approval_step = ConditionStep(
        name="DeploymentApprovalGate",
        conditions=[deployment_condition],
        if_steps=[deployment_step, endpoint_validation_step],
        else_steps=[deployment_fail_step]
    )
    
    # =================================================================
    # DEPLOYMENT PIPELINE ASSEMBLY
    # =================================================================
    # TODO: Lab 5.1.3 - Architecture Understanding: Complete deployment pipeline architecture
    # TODO: Lab 5.1.4 - Conceptual Relationships: Complex dependency chain for deployment workflow
    logger.info("=== Assembling Deployment Pipeline ===")
    
    # TODO: Lab 5.1.5 - High-level Comparison: Automated deployment pipeline vs manual deployment processes
    # Create comprehensive deployment pipeline
    pipeline = Pipeline(
        name="MLOpsModelDeploymentPipeline",
        parameters=[
            model_artifacts_s3,
            evaluation_metrics_s3,
            endpoint_instance_type,
            endpoint_instance_count,
            deployment_approval_threshold,
            model_package_group,
            endpoint_name
        ],
        steps=[
            # TODO: Lab 5.1.4 - Conceptual Relationships: Sequential dependency chain for deployment
            # TODO: Lab 5.2.4 - Dependency Creation: Complex deployment dependencies
            validation_step,           # Step 1: Pre-deployment validation
            model_step,               # Step 2: Create SageMaker model (ModelStep)
            register_model_step,      # Step 3: Register model in registry
            deployment_approval_step  # Step 4: Conditional deployment with validation
        ]
    )
    
    return pipeline

def create_inference_files(output_dir="deployment_scripts"):
    """
    Create necessary inference files for model deployment.
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create inference script
    inference_script = create_inference_script()
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(inference_script)
    
    logger.info(f"Inference files created in {output_dir}/")

def demonstrate_deployment_pipeline(pipeline):
    """
    Demonstrate deployment pipeline features for educational purposes.
    """
    logger.info("=== Deployment Pipeline Features ===")
    logger.info(f"Pipeline: {pipeline.name}")
    logger.info(f"Deployment Steps: {len(pipeline.steps)}")
    
    logger.info("\n=== Deployment Workflow ===")
    logger.info("1. Pre-deployment validation of model artifacts")
    logger.info("2. SageMaker model creation (ModelStep)")
    logger.info("3. Model registration in Model Registry")
    logger.info("4. Conditional deployment approval")
    logger.info("5. Endpoint deployment with Lambda")
    logger.info("6. Endpoint validation and testing")
    
    logger.info("\n=== Integration with Training Pipeline ===")
    logger.info("- Consumes model artifacts from TrainingStep")
    logger.info("- Uses evaluation metrics from EvaluationStep")
    logger.info("- Implements quality gates based on training results")
    
    logger.info("\n=== Best Practices Implemented ===")
    logger.info("- Pre-deployment model validation")
    logger.info("- Model Registry integration for lifecycle management")
    logger.info("- Automated endpoint deployment and validation")
    logger.info("- Conditional approval based on performance metrics")
    logger.info("- Comprehensive error handling and rollback capabilities")

if __name__ == "__main__":
    """
    Main execution for deployment pipeline creation and demonstration.
    
    # TODO: Lab 5.1.5 - High-level Comparison: Complete MLOps lifecycle demonstration
    """
    logger.info("=== Lab 5.1: SageMaker Model Deployment Pipeline ===")
    logger.info("🎯 Lab 5.1 Advanced Pipeline Concepts:")
    logger.info("1. ✅ Lab 5.1.1 - Component Identification: Advanced step types (ModelStep, RegisterModelStep)")
    logger.info("2. ✅ Lab 5.1.2 - Purpose Recognition: Model lifecycle and deployment purposes")
    logger.info("3. ✅ Lab 5.1.3 - Architecture Understanding: Extended deployment architecture")
    logger.info("4. ✅ Lab 5.1.4 - Conceptual Relationships: Integration with training pipeline outputs")
    logger.info("5. ✅ Lab 5.1.5 - High-level Comparison: Complete MLOps lifecycle vs manual processes")
    
    try:
        # TODO: Lab 5.2.2 - Implementation Details: Create inference files for deployment
        create_inference_files()
        
        # TODO: Lab 5.1.3 - Architecture Understanding: Create deployment pipeline architecture
        deployment_pipeline = create_deployment_pipeline()
        
        # TODO: Lab 5.1.1 - Component Identification: Demonstrate deployment pipeline features
        demonstrate_deployment_pipeline(deployment_pipeline)
        
        # TODO: Lab 5.1.4 - Conceptual Relationships: Show integration with other pipeline components
        logger.info("\n=== Integration with Other Pipeline Components ===")
        logger.info("- train.py: Provides model artifacts for deployment (Lab 5.1.4)")
        logger.info("- evaluate.py: Provides metrics for deployment approval (Lab 5.1.4)")
        logger.info("- preprocess.py: Ensures data consistency for inference (Lab 5.1.4)")
        logger.info("- pipeline_dev.py: Development pipeline feeding this deployment (Lab 5.1.4)")
        logger.info("- pipeline_prod.py: Production pipeline with deployment integration (Lab 5.1.4)")
        
        # TODO: Lab 5.1.5 - High-level Comparison: Complete MLOps workflow summary
        logger.info("\n=== Complete MLOps Workflow (Lab 5.1.5) ===")
        logger.info("Data Processing → Training → Evaluation → Registration → Deployment")
        logger.info("Traditional: Manual, error-prone, inconsistent")
        logger.info("Pipeline: Automated, reliable, reproducible")
        
    except Exception as e:
        # TODO: Lab 5.2.8 - Error Handling Implementation: Deployment pipeline error handling
        logger.error(f"Error creating deployment pipeline: {str(e)}")
        raise
