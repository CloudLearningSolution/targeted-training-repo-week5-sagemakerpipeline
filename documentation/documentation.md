---
challenge:
    module: Convert a notebook to production code
    challenge: '0: Convert a notebook to production code'
---

<style>
.button  {
  border: none;
  color: white;
  padding: 12px 28px;
  background-color: #4285F4;
  float: right;
}
</style>

# Challenge 0: Convert a notebook to production code

<button class="button" onclick="window.location.href='https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform';">Back to overview</button>

## Challenge scenario

The first step to automate machine learning workflows is to convert a Jupyter notebook to production-ready code. When you store your code as scripts, it's easier to automate the code execution. You can parameterize scripts to easily reuse the code for retraining with Google Cloud Vertex AI.

## Prerequisites

To complete this challenge, you'll need:

- Access to a Google Cloud Platform (GCP) account with appropriate permissions.
- A GitHub account.
- Basic familiarity with Vertex AI and Google Cloud services.
- Google Cloud SDK (gcloud) installed and configured.

## Objectives

By completing this challenge, you'll learn how to:

- Clean nonessential code.
- Convert your code to Python scripts compatible with Vertex AI.
- Use functions in your scripts.
- Use parameters in your scripts.
- Implement experiment tracking with Vertex AI Experiments.

> **Important!**
> Each challenge is designed to allow you to explore how to implement DevOps principles when working with machine learning models on Google Cloud Platform. Some instructions may be intentionally vague, inviting you to think about your own preferred approach. If for example, the instructions ask you to create a Vertex AI Workbench instance or enable Vertex AI APIs, it's up to you to explore and decide how you want to create it. To make it the best learning experience for you, it's up to you to make it as simple or as challenging as you want.

## Challenge Duration

- **Estimated Time**: 30 minutes

## Instructions

To work through the challenges, you need **your own public repo** which includes the challenge files. Create a new public repo by navigating to [https://github.com/GoogleCloudPlatform/vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples) and fork or use as a template to create your own repo.

In the **experimentation** folder, you'll find a Jupyter notebook that trains a classification model. The data used by the notebook is in the **experimentation/data** folder and contains a CSV file.

In the **src/model** folder you'll find a `train.py` script which already includes code converted from part of the notebook. It's up to you to complete it for Vertex AI compatibility.

- Go through the notebook to understand what the code does.
- Convert the code under the **Split data** header and include it in the `train.py` script as a `split_data` function. Remember to:
    - Remove nonessential code.
    - Include the necessary code as a function.
    - Include any necessary libraries at the top of the script.
    - Ensure compatibility with Vertex AI training environment.

<details>
<summary>Hint</summary>
<br/>
The <code>split_data</code> function is already included in the main function. You only need to add the function itself with the required inputs and outputs underneath the comment <code>TO DO: add function to split data</code>. Make sure to handle Vertex AI's expected input/output paths using environment variables like <code>AIP_MODEL_DIR</code> and <code>AIP_TRAINING_DATA_URI</code>.
</details>

- Add experiment tracking so that every time you run the script, all parameters and metrics are tracked. Use Vertex AI Experiments to track your training runs, or alternatively, integrate with Vertex AI's managed MLflow to ensure the necessary model files are stored with the job run for easy deployment.

<details>
<summary>Hint</summary>
<br/>
Vertex AI provides native experiment tracking capabilities through Vertex AI Experiments. You can also use the managed MLflow service on Vertex AI for experiment tracking. For Vertex AI Experiments, use the Vertex AI SDK to create and track experiments. For MLflow integration, you can use <code>mlflow.autolog()</code> with Vertex AI's managed MLflow tracking server. Enable experiment tracking in the main function under <code>TO DO: enable experiment tracking</code>.
</details>

- Ensure your script is compatible with Vertex AI custom training jobs by:
    - Using Vertex AI environment variables for input/output paths
    - Saving the model to the correct output directory (using `AIP_MODEL_DIR`)
    - Adding proper argument parsing for hyperparameters
    - Integrating with Google Cloud Storage for data access

<details>
<summary>Hint</summary>
<br/>
Vertex AI provides specific environment variables like <code>AIP_MODEL_DIR</code>, <code>AIP_TRAINING_DATA_URI</code>, and <code>AIP_VALIDATION_DATA_URI</code>. Use these to make your script portable across different Vertex AI environments. Also, implement argument parsing using <code>argparse</code> to handle hyperparameters passed from Vertex AI training jobs. Use the Google Cloud Storage client library to read training data from GCS buckets.
</details>

- Integrate with Google Cloud services:
    - Use Google Cloud Storage for data storage and model artifacts
    - Implement proper logging with Google Cloud Logging
    - Consider using Vertex AI Pipelines for workflow orchestration

<details>
<summary>Hint</summary>
<br/>
Import the necessary Google Cloud libraries: <code>google-cloud-storage</code> for GCS operations, <code>google-cloud-logging</code> for structured logging, and <code>google-cloud-aiplatform</code> for Vertex AI integration. Set up proper authentication using Application Default Credentials (ADC) or service account keys.
</details>

## Success criteria

To complete this challenge successfully, you should be able to show:

- A training script which includes a function to split the data and experiment tracking using Vertex AI Experiments or managed MLflow.
- The script is compatible with Vertex AI custom training jobs (uses appropriate environment variables and paths).
- Proper model serialization and saving to Google Cloud Storage.
- Integration with Google Cloud services for logging and data management.

> **Note:**
> If you've used a Vertex AI Workbench instance or Colab Enterprise for experimentation, remember to stop the instance when you're done to avoid unnecessary charges. Also, clean up any Google Cloud Storage buckets and Vertex AI resources you created during testing.

## Useful resources

- [Vertex AI Custom Training Documentation](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Vertex AI Experiments for Experiment Tracking](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)
- [Using MLflow with Vertex AI](https://cloud.google.com/vertex-ai/docs/experiments/vertex-ai-mlflow)
- [Vertex AI Workbench User Guide](https://cloud.google.com/vertex-ai/docs/workbench)
- [Google Cloud Storage Client Libraries](https://cloud.google.com/storage/docs/reference/libraries)
- [Vertex AI Python SDK Documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest)
- [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
- [Google Cloud ML Engineering Best Practices](https://cloud.google.com/architecture/ml-on-gcp-best-practices)

<button class="button" onclick="window.location.href='01-vertex-ai-job';">Continue with challenge 1</button>
