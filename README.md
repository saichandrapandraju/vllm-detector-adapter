# VLLM Detector Adapter

A model-agnostic adapter for enabling LLM-as-a-judge detections and evaluations with vLLM-hosted models.

## Overview

The `vllm-detector-adapter` is a standalone microservice that sits between client applications and the vLLM server, providing a specialized API for using vLLM-hosted language models as judges or evaluators. The adapter enables any arbitrary LLM hosted on vLLM to function as an evaluator without requiring model-specific hardcoding.

## Features

- **Model-Agnostic Design**: Works with any LLM hosted on vLLM
- **Flexible Prompt Templates**: Configurable templates optimized for different judge LLMs and evaluation tasks
- **Comprehensive API**: Endpoints for single text evaluation, pairwise comparison, and template management
- **Robust Output Parsing**: Intelligently extracts structured judgments from LLM responses
- **Asynchronous Processing**: Non-blocking API design for handling evaluation tasks

## Architecture

The adapter follows a modular design with these key components:

```
Client Application → vllM-Detector-Adapter → vLLM Server (hosting judge LLMs)
```

- **API Layer**: FastAPI application defining the adapter's endpoints
- **Service Layer**: Core components handling prompt management, vLLM communication, and output parsing
- **Configuration Layer**: Settings and models for the adapter

## Installation

### Prerequisites

- Python 3.9+
- Access to a running vLLM server

### Using Docker

```bash
# Build the Docker image
docker build -t vllm-detector-adapter .

# Run the container
docker run -p 8000:8000 \
  -e VLLM_API_BASE=http://your-vllm-server:8000/v1 \
  -e VLLM_API_KEY=your-api-key \
  vllm-detector-adapter
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vllm-detector-adapter.git
cd vllm-detector-adapter

# Install dependencies
pip install -r requirements.txt

# Run the application
VLLM_API_BASE=http://your-vllm-server:8000/v1 VLLM_API_KEY=your-api-key uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

The adapter can be configured using the following environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `VLLM_API_BASE` | Base URL of the vLLM server | `http://vllm-server:8000/v1` |
| `VLLM_API_KEY` | API key for the vLLM server (if needed) | `None` |
| `DEFAULT_TIMEOUT` | Timeout for requests to vLLM server (in seconds) | `60` |
| `MAX_RETRY_ATTEMPTS` | Maximum number of retry attempts for failed requests | `3` |
| `TEMPLATE_STORAGE_PATH` | Path to the template storage file | `app/templates/default_templates.json` |
| `TASK_EXPIRY_SECONDS` | Time until completed tasks are removed from memory (in seconds) | `3600` |

## API Reference

### Evaluation Endpoints

#### Evaluate a Single Text

```
POST /v1/evaluate/single_response
```

Request body:

```json
{
  "judge_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "text_to_evaluate": "This product is amazing and I love it!",
  "evaluation_criteria": "Determine if the text expresses a positive or negative sentiment.",
  "prompt_template_id": "binary_classification",
  "custom_prompt_segments": {
    "system_message": "You are a sentiment analysis expert.",
    "user_instruction_prefix": "Analyze the sentiment of this text:\n\n",
    "user_instruction_suffix": "\n\nProvide your analysis:"
  },
  "output_format_instruction": "Respond with ONLY 'POSITIVE' or 'NEGATIVE'.",
  "vllm_sampling_params": {
    "max_tokens": 100,
    "temperature": 0.2
  },
  "provide_reasoning": true
}
```

#### Compare Two Texts

```
POST /v1/evaluate/pairwise_comparison
```

Request body:

```json
{
  "judge_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "text_A": "The product arrived on time and worked as expected.",
  "text_B": "The product was delivered promptly and functioned perfectly. It exceeded my expectations in terms of quality and performance.",
  "comparison_criteria": "Compare these two product reviews based on detail and helpfulness.",
  "output_format_instruction": "Respond with 'A' if the first review is better, 'B' if the second review is better, or 'EQUAL' if they are of equal quality.",
  "provide_reasoning": true
}
```

#### Check Evaluation Status

```
GET /v1/evaluate/status/{evaluation_id}
```

### Configuration Endpoints

#### List Prompt Templates

```
GET /v1/config/judge_templates
```

#### Get a Specific Template

```
GET /v1/config/judge_templates/{template_id}
```

#### Create a New Template

```
POST /v1/config/judge_templates
```

#### Update a Template

```
PUT /v1/config/judge_templates/{template_id}
```

#### Delete a Template

```
DELETE /v1/config/judge_templates/{template_id}
```

## Usage Examples

### Python Client Example

```python
import requests
import time
import json

# Base URL of the adapter
BASE_URL = "http://localhost:8000/v1"

# Example 1: Evaluate a single text
def evaluate_single_text():
    response = requests.post(
        f"{BASE_URL}/evaluate/single_response",
        json={
            "judge_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "text_to_evaluate": "This product is amazing and I love it!",
            "evaluation_criteria": "Determine if the text expresses a positive or negative sentiment.",
            "output_format_instruction": "Respond with ONLY 'POSITIVE' or 'NEGATIVE'.",
            "provide_reasoning": True
        }
    )
    
    if response.status_code == 200:
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        print(f"Started evaluation task: {evaluation_id}")
        
        # Wait for the task to complete
        result = None
        while result is None:
            status_response = requests.get(f"{BASE_URL}/evaluate/status/{evaluation_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "COMPLETED":
                result = status_data["result"]
                print(f"Judgment: {result['judgment']}")
                if result.get("reasoning"):
                    print(f"Reasoning: {result['reasoning']}")
                break
            elif status_data["status"] == "FAILED":
                print(f"Evaluation failed: {status_data['error_message']}")
                break
            
            print("Waiting for evaluation to complete...")
            time.sleep(1)
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example 2: Pairwise comparison
def compare_texts():
    response = requests.post(
        f"{BASE_URL}/evaluate/pairwise_comparison",
        json={
            "judge_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "text_A": "The product arrived on time and worked as expected.",
            "text_B": "The product was delivered promptly and functioned perfectly. It exceeded my expectations in terms of quality and performance.",
            "comparison_criteria": "Compare these two product reviews based on detail and helpfulness.",
            "output_format_instruction": "Respond with 'A' if the first review is better, 'B' if the second review is better, or 'EQUAL' if they are of equal quality.",
            "provide_reasoning": True
        }
    )
    
    if response.status_code == 200:
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        print(f"Started comparison task: {evaluation_id}")
        
        # Wait for the task to complete
        result = None
        while result is None:
            status_response = requests.get(f"{BASE_URL}/evaluate/status/{evaluation_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "COMPLETED":
                result = status_data["result"]
                print(f"Preferred text: {result['judgment']}")
                if result.get("reasoning"):
                    print(f"Reasoning: {result['reasoning']}")
                break
            elif status_data["status"] == "FAILED":
                print(f"Comparison failed: {status_data['error_message']}")
                break
            
            print("Waiting for comparison to complete...")
            time.sleep(1)
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example 3: Create a custom prompt template
def create_template():
    response = requests.post(
        f"{BASE_URL}/config/judge_templates",
        json={
            "template_name": "Toxicity Detection",
            "target_judge_model_family": "Mistral",
            "description": "Template for detecting toxic content in text",
            "prompt_structure": {
                "system_message": "You are a content moderation expert. Your task is to detect toxic content in the provided text.",
                "user_instruction_prefix": "Please analyze the following text for toxic content based on these criteria:\n\n{evaluation_criteria}\n\nText to analyze:\n\n",
                "user_instruction_suffix": "\n\n{output_format_instruction}"
            },
            "output_parser_rules": {
                "type": "binary",
                "positive_patterns": ["toxic", "inappropriate", "harmful"],
                "negative_patterns": ["non-toxic", "appropriate", "safe"]
            }
        }
    )
    
    if response.status_code == 200:
        template = response.json()
        print(f"Created template: {template['template_id']} - {template['template_name']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Run the examples
if __name__ == "__main__":
    evaluate_single_text()
    compare_texts()
    create_template()
```

## Testing

Run the test suite with:

```bash
python -m pytest
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
