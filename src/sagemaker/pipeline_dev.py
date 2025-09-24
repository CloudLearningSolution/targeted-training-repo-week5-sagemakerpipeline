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
from sagemaker.workflow.pipeline_context import PipelineSession
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
    
    # TODO: Lab 5.1.3 - Architecture Understanding: Complete pipeline architecture demonstration
    # TODO: Lab 5.1.4 - Conceptual Relationships: ProcessingStep → TrainingStep → EvaluationStep → ConditionStep
    # TODO: Lab 5.1.5 - High-level Comparison: Pipeline architecture vs traditional sequential notebooks
    
    Args:
        role (str): SageMaker execution role ARN
        bucket_name (str): S3 bucket for pipeline artifacts
        region (str): AWS region
        
    Returns:
        Pipeline: Configured SageMaker Pipeline representing the component architecture
    """
    
    # Get execution role if not provided
    if role is None:
        try:
            role = get_execution_role()
            logger.info(f"Using execution role: {role}")
        except Exception:
            role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            logger.warning(f"Using default role: {role}")
    
    # Create PipelineSession for proper pipeline execution
    pipeline_session = PipelineSession()
    
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
    # STEP 1: DATA PROCESSING - ProcessingStep Component
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: ProcessingStep is core step type for data processing
    # TODO: Lab 5.1.2 - Purpose Recognition: Data preprocessing, validation, feature engineering
    # TODO: Lab 5.1.3 - Architecture Understanding: Often serves as starting component in ML pipelines
    logger.info("=== Defining ProcessingStep Component ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: Configure SKLearnProcessor for ProcessingStep
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="mlops-data-preprocessing",
        sagemaker_session=pipeline_session
    )
    
    # TODO: Lab 5.1.4 - Conceptual Relationships: ProcessingStep creates outputs for downstream components
    # TODO: Lab 5.2.2 - Implementation Details: Configure ProcessingStep inputs and outputs
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
            # TODO: Lab 5.2.3 - Property References: Output names used for step property references
            ProcessingOutput(
                output_name="train_data",  # Referenced by TrainingStep
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket_name}/pipeline-dev/train/"
            ),
            ProcessingOutput(
                output_name="test_data",   # Referenced by EvaluationStep
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket_name}/pipeline-dev/test/"
            )
        ],
        arguments=[
            "--test_size", test_size
        ]
    )
    
    # =================================================================
    # STEP 2: MODEL TRAINING - TrainingStep Component
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: TrainingStep is core step type for model training
    # TODO: Lab 5.1.2 - Purpose Recognition: Model training, hyperparameter optimization, ML algorithms
    # TODO: Lab 5.1.4 - Conceptual Relationships: Depends on ProcessingStep for training data
    logger.info("=== Defining TrainingStep Component ===")
    
    # TODO: Lab 5.2.5 - TrainingStep Configuration: Configure SKLearn estimator for TrainingStep
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
        base_job_name="mlops-model-training",
        sagemaker_session=pipeline_session
    )
    
    # TODO: Lab 5.1.4 - Conceptual Relationships: TrainingStep input references ProcessingStep output
    # TODO: Lab 5.2.3 - Property References: Use property references to create component dependencies
    # TODO: Lab 5.2.4 - Dependency Creation: Create step-to-step dependencies using property references
    training_step = TrainingStep(
        name="ModelTrainingStep",
        step_args=sklearn_estimator.fit(
            inputs={
                "train": TrainingInput(
                    # TODO: Lab 5.2.3 - Property References: Reference ProcessingStep output
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
    )
    
    # =================================================================
    # STEP 3: MODEL EVALUATION - EvaluationStep Component (ProcessingStep)
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: EvaluationStep uses ProcessingStep for evaluation
    # TODO: Lab 5.1.2 - Purpose Recognition: ProcessingStep flexibility for different ML tasks
    # TODO: Lab 5.1.4 - Conceptual Relationships: Depends on BOTH ProcessingStep AND TrainingStep
    logger.info("=== Defining EvaluationStep Component ===")
    
    # TODO: Lab 5.2.1 - Step Configuration: Configure ProcessingStep for evaluation workload
    # TODO: Lab 5.2.2 - Implementation Details: Configure EvaluationStep inputs and outputs
    evaluation_step = ProcessingStep(
        name="ModelEvaluationStep",
        processor=sklearn_processor,
        code="evaluate.py",
        inputs=[
            # TODO: Lab 5.1.4 - Conceptual Relationships: Multiple component dependencies
            # TODO: Lab 5.2.3 - Property References: Reference TrainingStep model artifacts
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model"
            ),
            # TODO: Lab 5.2.3 - Property References: Reference ProcessingStep test data
            # TODO: Lab 5.2.4 - Dependency Creation: Create multiple input dependencies
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
        # TODO: Lab 5.2.3 - Property References: PropertyFile enables ConditionStep to read evaluation results
        property_files=[
            PropertyFile(
                name="EvaluationReport",
                output_name="evaluation_metrics", 
                path="evaluation.json"
            )
        ]
    )
    
    # =================================================================
    # STEP 4: CONDITIONAL LOGIC - ConditionStep Component
    # =================================================================
    # TODO: Lab 5.1.1 - Component Identification: ConditionStep is core step type for conditional logic
    # TODO: Lab 5.1.2 - Purpose Recognition: Quality gates, approval workflows, conditional branching
    # TODO: Lab 5.1.4 - Conceptual Relationships: Depends on EvaluationStep for decision data
    logger.info("=== Defining ConditionStep Component ===")
    
    # =================================================================
    # OPTIONAL EXTENSION: TRANSFORM STEP - BatchTransform Component
    # =================================================================
    # TODO: Lab 5.2.7 - Transform Step Usage: Implement batch transform jobs (optional extension)
    # TODO: How to scaffold a TransformStep using model artifact from TrainingStep:
    # TODO: from sagemaker.transformer import Transformer
    # TODO: transformer = Transformer(
    # TODO:     model_name=training_step.properties.ModelName,
    # TODO:     instance_count=1,
    # TODO:     instance_type="ml.m5.large",
    # TODO:     output_path=f"s3://{bucket_name}/transform-output/"
    # TODO: )
    # TODO: transform_step = TransformStep(
    # TODO:     name="BatchTransformStep", 
    # TODO:     transformer=transformer,
    # TODO:     inputs=TransformInput(data=f"s3://{bucket_name}/batch-data/")
    # TODO: )
    # TODO: This is an optional extension - not implemented in current pipeline
    
    # TODO: Lab 5.1.2 - Purpose Recognition: ConditionStep reads evaluation results for decisions
    # TODO: Lab 5.2.3 - Property References: Access evaluation metrics via PropertyFile reference
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=evaluation_step.properties.PropertyFiles.EvaluationReport.JsonGet("accuracy"),
        right=min_accuracy_threshold
    )
    
    # Fail step for when model doesn't meet criteria
    fail_step = FailStep(
        name="ModelFailureStep",
        error_message="Model accuracy below threshold. Pipeline terminated."
    )
    
    # TODO: Lab 5.1.2 - Purpose Recognition: ConditionStep creates branching logic in pipeline architecture
    # TODO: Lab 5.1.5 - High-level Comparison: Conditional pipeline logic vs manual notebook decisions
    condition_step = ConditionStep(
        name="ModelApprovalCondition",
        conditions=[accuracy_condition],
        if_steps=[],  # Could add deployment steps here
        else_steps=[fail_step]  # Fail pipeline if model doesn't meet criteria
    )
    
    # =================================================================
    # PIPELINE ASSEMBLY - Complete SageMaker Pipeline Architecture
    # =================================================================
    # TODO: Lab 5.1.3 - Architecture Understanding: Pipeline object represents complete component architecture
    # TODO: Lab 5.1.4 - Conceptual Relationships: Components connected through dependencies, not execution order
    logger.info("=== Assembling Complete Pipeline Architecture ===")
    
    # TODO: Lab 5.1.3 - Architecture Understanding: Final component architecture summary
    # TODO: Lab 5.2.4 - Dependency Creation: SageMaker calculates execution order from dependencies
    
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
            # TODO: Lab 5.1.4 - Conceptual Relationships: Step order in list doesn't determine execution order
            # TODO: Lab 5.2.4 - Dependency Creation: SageMaker Pipeline SDK calculates execution order from dependencies
            processing_step,    # ProcessingStep: Data preprocessing component
            training_step,      # TrainingStep: Model training component  
            evaluation_step,    # EvaluationStep: Model evaluation component
            condition_step      # ConditionStep: Conditional approval component
        ],
        sagemaker_session=pipeline_session  # Use pipeline session
    )
    
    return pipeline

def demonstrate_pipeline_properties(pipeline):
    """
    Demonstrate key pipeline properties for educational purposes.
    Shows students how to inspect pipeline structure and dependencies.
    
    # TODO: Lab 5.1 - Analyzing Pipeline Component Architecture
    # TODO: This function helps participants understand the pipeline architecture they built
    # TODO: Focus on component types, relationships, and architecture patterns
    """
    logger.info("=== Lab 5.1: Pipeline Architecture Analysis ===")
    logger.info(f"Pipeline Name: {pipeline.name}")
    logger.info(f"Number of Components: {len(pipeline.steps)}")
    logger.info(f"Number of Parameters: {len(pipeline.parameters)}")
    
    # TODO: Lab 5.1 - Component Types and Purposes Analysis
    # TODO: Identify and categorize the core SageMaker Pipeline step types
    logger.info("\n=== Core SageMaker Pipeline Step Types ===")
    logger.info("🔍 Step Type Analysis:")
    
    for step in pipeline.steps:
        logger.info(f"\nComponent: {step.name}")
        logger.info(f"  Step Type: {type(step).__name__}")
        
        # TODO: Lab 5.1 - Step Type Purposes
        # TODO: Explain the purpose of each core step type
        if isinstance(step, ProcessingStep):
            if "Preprocessing" in step.name:
                logger.info(f"  Purpose: Data preprocessing and validation")
            elif "Evaluation" in step.name:
                logger.info(f"  Purpose: Model evaluation and metrics generation")
        elif isinstance(step, TrainingStep):
            logger.info(f"  Purpose: Model training and hyperparameter optimization")
        elif isinstance(step, ConditionStep):
            logger.info(f"  Purpose: Conditional logic and quality gates")
    
    # TODO: Lab 5.1 - Component Relationships Overview
    # TODO: Show conceptual relationships between components
    logger.info("\n=== Component Relationships (Architecture Overview) ===")
    logger.info("📊 Pipeline Architecture Pattern:")
    logger.info("   ProcessingStep (data prep) → TrainingStep (model training)")
    logger.info("   ProcessingStep (data prep) → EvaluationStep (model evaluation)")  
    logger.info("   TrainingStep (model training) → EvaluationStep (model evaluation)")
    logger.info("   EvaluationStep (model evaluation) → ConditionStep (approval gate)")
    
    # TODO: Lab 5.1 - vs Traditional ML Workflows
    # TODO: Compare pipeline architecture to traditional ML approaches
    logger.info("\n=== vs Traditional ML Workflow Patterns ===")
    logger.info("📈 Architecture Comparison:")
    logger.info("   Traditional ML:")
    logger.info("     • Monolithic notebooks with mixed concerns")
    logger.info("     • Manual execution order and dependency management")
    logger.info("     • Limited reusability and modularity")
    logger.info("")
    logger.info("   SageMaker Pipeline Architecture:")
    logger.info("     • Modular components with single responsibilities")
    logger.info("     • Automatic dependency resolution and execution order")
    logger.info("     • Reusable components and configurable workflows")
    
    # TODO: Lab 5.1 - Parameter-Driven Architecture
    logger.info("\n=== Parameter-Driven Architecture ===")
    logger.info("📊 Pipeline parameters enable flexible architecture:")
    for param in pipeline.parameters:
        logger.info(f"  {param.name}: {param.default_value}")
    
    # TODO: Lab 5.1 - Architecture Benefits Summary
    logger.info("\n=== SageMaker Pipeline Architecture Benefits ===")
    logger.info("✅ Modular Design: Each component has focused responsibility")
    logger.info("✅ Dependency Management: Automatic execution order resolution")  
    logger.info("✅ Reusability: Components can be reused across different pipelines")
    logger.info("✅ Scalability: Each component can scale independently")
    logger.info("✅ Maintainability: Clear separation of concerns")
    
    # TODO: Step 1.20: Detailed Step Dependency Analysis 
    logger.info("\n=== Step Dependencies (DAG Structure) ===")
    logger.info("🔗 Analyzing step-to-step dependencies:")
    
    for step in pipeline.steps:
        logger.info(f"\nStep: {step.name}")
        logger.info(f"  Type: {type(step).__name__}")
        
        # Analyze inputs for dependencies
        if hasattr(step, 'inputs') and step.inputs:
            logger.info(f"  Dependencies created through inputs:")
            for input_item in step.inputs:
                if hasattr(input_item, 'source'):
                    logger.info(f"    - {input_item.source}")
                    if 'properties' in str(input_item.source):
                        logger.info(f"      💡 This creates a DAG dependency!")
    
    # TODO: Step 1.21: Parameter Analysis for DAG Configuration  
    logger.info("\n=== Parameter Configuration (DAG Flexibility) ===")
    logger.info("📊 Parameters make the DAG reusable with different configurations:")
    for param in pipeline.parameters:
        logger.info(f"  {param.name}: {param.default_value}")
    
    # TODO: Step 1.22: DAG Validation Checklist
    logger.info("\n=== DAG Validation Checklist ===")
    logger.info("✅ DIRECTED: Each step has clear input → output flow")
    logger.info("✅ ACYCLIC: No step depends on its own outputs (no cycles)")  
    logger.info("✅ GRAPH: Steps are interconnected through property references")
    logger.info("✅ EXECUTABLE: Dependencies ensure proper execution order")

if __name__ == "__main__":
    """
    Main execution block for creating and analyzing the development pipeline.
    This demonstrates the complete pipeline creation process for educational purposes.
    
    # TODO: Lab 5.1.3 - Architecture Understanding: Complete pipeline component architecture overview
    # TODO: Lab 5.1.5 - High-level Comparison: This section demonstrates all Lab 5.1 learning objectives
    """
    logger.info("=== Lab 5.1: SageMaker Pipeline Component Architecture Overview ===")
    logger.info("🎯 Lab 5.1 Learning Objectives:")
    logger.info("1. ✅ Lab 5.1.1 - Component Identification: Core SageMaker Pipeline step types")
    logger.info("2. ✅ Lab 5.1.2 - Purpose Recognition: Understanding step type purposes") 
    logger.info("3. ✅ Lab 5.1.3 - Architecture Understanding: Pipeline component architecture")
    logger.info("4. ✅ Lab 5.1.4 - Conceptual Relationships: How components connect conceptually")
    logger.info("5. ✅ Lab 5.1.5 - High-level Comparison: Pipeline vs traditional ML workflows")
    logger.info("")
    
    # TODO: Lab 5.1.3 - Architecture Understanding: SageMaker Pipeline SDK architecture concepts
    logger.info("🏗️ SageMaker Pipeline SDK Component Architecture:")
    logger.info("   📍 COMPONENT-BASED DESIGN:")
    logger.info("      • Modular components with focused responsibilities")
    logger.info("      • Reusable components across different pipelines")
    logger.info("      • Clear separation of concerns")
    logger.info("")
    logger.info("   📊 STEP TYPE IDENTIFICATION:")
    logger.info("      • ProcessingStep: Data processing workloads")
    logger.info("      • TrainingStep: Model training workloads") 
    logger.info("      • ConditionStep: Conditional logic and quality gates")
    logger.info("")
    
    try:
        # TODO: Lab 5.1.3 - Architecture Understanding: Create complete pipeline architecture
        logger.info("🔨 Creating SageMaker Pipeline component architecture...")
        dev_pipeline = create_development_pipeline()
        
        # TODO: Lab 5.1.1 - Component Identification: Analyze created pipeline architecture
        logger.info("🔍 Analyzing pipeline component architecture...")
        demonstrate_pipeline_properties(dev_pipeline)
        
        # TODO: Lab 5.1.5 - High-level Comparison: Architecture benefits summary
        logger.info("\n=== Lab 5.1 Component Architecture Summary ===")
        logger.info("💡 Key Architectural Concepts Demonstrated:")
        logger.info("1. 📊 Lab 5.1.1 - Core Step Types: ProcessingStep, TrainingStep, ConditionStep")
        logger.info("2. 🔗 Lab 5.1.4 - Component Relationships: Dependencies through property references")
        logger.info("3. 🏗️ Lab 5.1.3 - Architecture Design: Modular component-based architecture")
        logger.info("4. 📈 Lab 5.1.5 - vs Traditional: Component architecture vs monolithic notebooks")
        
        # TODO: Lab 5.1.5 - High-level Comparison: Lab completion and next steps
        logger.info("\n=== Lab 5.1 Completion ===")
        logger.info("🎉 Pipeline Component Architecture Overview completed!")
        logger.info("📋 Lab 5.1 Understanding Achieved:")
        logger.info("   • Component identification and categorization")
        logger.info("   • Step type purposes and responsibilities")
        logger.info("   • Pipeline architecture patterns")
        logger.info("   • Component relationships and conceptual understanding")
        logger.info("   • Architectural benefits vs traditional ML workflows")
        logger.info("")
        logger.info("🚀 Ready for Lab 5.2: SageMaker Processing, Training, and Transform Steps")
        
    except Exception as e:
        # TODO: Lab 5.2.8 - Error Handling Implementation: Pipeline architecture error handling
        logger.error(f"❌ Error in Lab 5.1 pipeline architecture: {str(e)}")
        logger.error("💡 Review component definitions and architecture patterns")
        raise
