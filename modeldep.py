import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# IAM role that SageMaker can assume to access S3
role = "arn:aws:iam::339713052627:role/service-role/SuperAdmin"

# Define the S3 path for your model artifacts (config.json, model.safetensors, optimizer.pt in .tar.gz)
model_s3_path = "https://mlgabifybucket.s3.amazonaws.com/model.tar.gz"

# Define the Hugging Face model configuration (using the pre-trained processor)
huggingface_model = HuggingFaceModel(
    model_data=model_s3_path,  # Path to your model artifacts
    role=role,
    transformers_version="4.6",  # Version of Hugging Face Transformers
    pytorch_version="1.7",       # PyTorch version
    py_version="py36"            # Python version
)

# Deploy the model as an endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,          # Number of instances
    instance_type="ml.m5.large"        # Type of EC2 instance
)
print("running")

# Sample input data (use appropriate input format for your model)
data = {
    "inputs": "This is a sample input text for Hugging Face model deployment."  # Example text input
}

# Use the deployed endpoint to get predictions
result = predictor.predict(data)

# Print the result
print(result)
