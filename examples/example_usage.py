import requests
import time
import json
from typing import Dict, Any, Optional


class VLLMJudgeClient:
    """Client for interacting with the vLLM-detector-adapter."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the adapter
        """
        self.base_url = base_url
    
    def evaluate_toxicity(
        self,
        text: str,
        judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        provide_reasoning: bool = True,
        wait_for_result: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate text for toxic content.
        
        Args:
            text: Text to evaluate
            judge_model_id: ID of the judge model
            provide_reasoning: Whether to request reasoning
            wait_for_result: Whether to wait for the result or return immediately
            timeout: Timeout in seconds (only used if wait_for_result is True)
            
        Returns:
            Evaluation response
        """
        response = requests.post(
            f"{self.base_url}/evaluate/single_response",
            json={
                "judge_model_id": judge_model_id,
                "text_to_evaluate": text,
                "evaluation_criteria": "Evaluate whether the text contains toxic, harmful, offensive, or inappropriate content that could be damaging to individuals or groups. Consider aspects such as hate speech, profanity, violent language, discriminatory statements, and personal attacks.",
                "prompt_template_id": "toxicity_detection",
                "output_format_instruction": "Respond with ONLY 'TOXIC' if the content contains any toxic elements, or 'NON-TOXIC' if it does not.",
                "provide_reasoning": provide_reasoning
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        if not wait_for_result:
            return task_data
        
        # Wait for the result
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise Exception(f"Evaluation failed: {status_data.get('error_message')}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
    
    def evaluate_factual_accuracy(
        self,
        text: str,
        reference_info: str,
        judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        wait_for_result: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of text against reference information.
        
        Args:
            text: Text to evaluate
            reference_info: Reference information to check against
            judge_model_id: ID of the judge model
            wait_for_result: Whether to wait for the result or return immediately
            timeout: Timeout in seconds (only used if wait_for_result is True)
            
        Returns:
            Evaluation response
        """
        response = requests.post(
            f"{self.base_url}/evaluate/single_response",
            json={
                "judge_model_id": judge_model_id,
                "text_to_evaluate": text,
                "evaluation_criteria": reference_info,
                "prompt_template_id": "factual_accuracy",
                "output_format_instruction": "Respond with JSON in this format: {\"accuracy_score\": <1-5>, \"errors_found\": [<list of factual errors>], \"is_accurate\": <true|false>}",
                "provide_reasoning": False,
                "vllm_sampling_params": {
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        if not wait_for_result:
            return task_data
        
        # Wait for the result
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise Exception(f"Evaluation failed: {status_data.get('error_message')}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
    
    def detect_hallucinations(
        self,
        generated_text: str,
        source_info: str,
        judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        wait_for_result: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in generated text compared to source information.
        
        Args:
            generated_text: Text to evaluate for hallucinations
            source_info: Source information to check against
            judge_model_id: ID of the judge model
            wait_for_result: Whether to wait for the result or return immediately
            timeout: Timeout in seconds (only used if wait_for_result is True)
            
        Returns:
            Evaluation response
        """
        response = requests.post(
            f"{self.base_url}/evaluate/single_response",
            json={
                "judge_model_id": judge_model_id,
                "text_to_evaluate": generated_text,
                "evaluation_criteria": source_info,
                "prompt_template_id": "hallucination_detection",
                "output_format_instruction": "Respond with JSON in this format: {\"contains_hallucinations\": <true|false>, \"hallucinated_claims\": [<list of hallucinated claims>], \"hallucination_severity\": <\"low\"|\"medium\"|\"high\">}",
                "provide_reasoning": False,
                "vllm_sampling_params": {
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        if not wait_for_result:
            return task_data
        
        # Wait for the result
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise Exception(f"Evaluation failed: {status_data.get('error_message')}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
    
    def evaluate_conversational_ai_response(
        self,
        user_query: str,
        ai_response: str,
        judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        wait_for_result: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate an AI assistant's response to a user query.
        
        Args:
            user_query: The user's query
            ai_response: The AI's response to evaluate
            judge_model_id: ID of the judge model
            wait_for_result: Whether to wait for the result or return immediately
            timeout: Timeout in seconds (only used if wait_for_result is True)
            
        Returns:
            Evaluation response
        """
        # Create a custom prompt template for this specific use case
        custom_prompt_segments = {
            "system_message": "You are an expert evaluator of conversational AI systems. Your task is to assess the quality of an AI assistant's response to a user query.",
            "user_instruction_prefix": "Please evaluate the quality of the following AI assistant's response to a user query. Consider aspects such as helpfulness, relevance, accuracy, and clarity.\n\nUser Query:\n\"\"\"\n{evaluation_criteria}\n\"\"\"\n\nAI Response to Evaluate:\n\"\"\"\n"
        }
        
        response = requests.post(
            f"{self.base_url}/evaluate/single_response",
            json={
                "judge_model_id": judge_model_id,
                "text_to_evaluate": ai_response,
                "evaluation_criteria": user_query,
                "prompt_template_id": "likert_scale",
                "custom_prompt_segments": custom_prompt_segments,
                "output_format_instruction": "Please provide your evaluation as JSON in this format: {\"overall_score\": <1-5>, \"helpfulness\": <1-5>, \"relevance\": <1-5>, \"accuracy\": <1-5>, \"clarity\": <1-5>, \"feedback\": \"<brief feedback>\"}",
                "provide_reasoning": False,
                "vllm_sampling_params": {
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        if not wait_for_result:
            return task_data
        
        # Wait for the result
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise Exception(f"Evaluation failed: {status_data.get('error_message')}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
    
    def compare_summarizations(
        self,
        original_text: str,
        summary_A: str,
        summary_B: str,
        judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        wait_for_result: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Compare two summarizations of the same original text.
        
        Args:
            original_text: The original text
            summary_A: First summary to compare
            summary_B: Second summary to compare
            judge_model_id: ID of the judge model
            wait_for_result: Whether to wait for the result or return immediately
            timeout: Timeout in seconds (only used if wait_for_result is True)
            
        Returns:
            Evaluation response
        """
        # Create custom prompt segments for this specific use case
        custom_prompt_segments = {
            "system_message": "You are an expert in content summarization. Your task is to compare two summaries of the same original text and determine which one is better.",
            "user_instruction_prefix": "Please compare the following two summaries of the original text. Evaluate which summary better captures the key information, is more concise, and is more accurate.\n\nOriginal Text:\n\"\"\"\n{comparison_criteria}\n\"\"\"\n\nSummary A:\n\"\"\"\n{text_A}\n\"\"\"\n\nSummary B:\n\"\"\"\n{text_B}\n\"\"\"\n\n"
        }
        
        response = requests.post(
            f"{self.base_url}/evaluate/pairwise_comparison",
            json={
                "judge_model_id": judge_model_id,
                "text_A": summary_A,
                "text_B": summary_B,
                "comparison_criteria": original_text,
                "prompt_template_id": "pairwise_comparison",
                "custom_prompt_segments": custom_prompt_segments,
                "output_format_instruction": "Respond with JSON in this format: {\"better_summary\": <\"A\"|\"B\"|\"EQUAL\">, \"reasoning\": \"<explanation>\", \"completeness\": {\"A\": <1-5>, \"B\": <1-5>}, \"conciseness\": {\"A\": <1-5>, \"B\": <1-5>}, \"accuracy\": {\"A\": <1-5>, \"B\": <1-5>}}",
                "provide_reasoning": False,
                "vllm_sampling_params": {
                    "max_tokens": 500,
                    "temperature": 0.1
                }
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        if not wait_for_result:
            return task_data
        
        # Wait for the result
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise Exception(f"Evaluation failed: {status_data.get('error_message')}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")
    
    def get_status(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get the status of an evaluation.
        
        Args:
            evaluation_id: ID of the evaluation
            
        Returns:
            Status response
        """
        response = requests.get(f"{self.base_url}/evaluate/status/{evaluation_id}")
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def create_custom_template(
        self,
        template_name: str,
        system_message: str,
        user_instruction_prefix: str,
        user_instruction_suffix: Optional[str] = None,
        output_parser_rules: Optional[Dict[str, Any]] = None,
        target_judge_model_family: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a custom prompt template.
        
        Args:
            template_name: Name of the template
            system_message: System message for the prompt
            user_instruction_prefix: Prefix for the user instruction
            user_instruction_suffix: Suffix for the user instruction
            output_parser_rules: Rules for parsing the output
            target_judge_model_family: Model family the template is optimized for
            description: Description of the template
            
        Returns:
            Created template
        """
        prompt_structure = {
            "system_message": system_message,
            "user_instruction_prefix": user_instruction_prefix,
        }
        
        if user_instruction_suffix:
            prompt_structure["user_instruction_suffix"] = user_instruction_suffix
        
        template_data = {
            "template_name": template_name,
            "prompt_structure": prompt_structure,
        }
        
        if output_parser_rules:
            template_data["output_parser_rules"] = output_parser_rules
        
        if target_judge_model_family:
            template_data["target_judge_model_family"] = target_judge_model_family
        
        if description:
            template_data["description"] = description
        
        response = requests.post(
            f"{self.base_url}/config/judge_templates",
            json=template_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = VLLMJudgeClient()
    
    # Example 1: Toxicity detection
    print("\n=== Example 1: Toxicity Detection ===")
    toxic_result = client.evaluate_toxicity(
        "This product is terrible and the customer service is even worse."
    )
    print(f"Judgment: {toxic_result['result']['judgment']}")
    if toxic_result['result'].get('reasoning'):
        print(f"Reasoning: {toxic_result['result']['reasoning']}")
    
    # Example 2: Factual accuracy check
    print("\n=== Example 2: Factual Accuracy Check ===")
    reference_info = """
    The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, the one-mile-wide strait connecting San Francisco Bay and the Pacific Ocean. The structure links the American city of San Francisco, California to Marin County. It was opened in 1937 and had the world's longest main span of 4,200 feet (1,280 m) until the Verrazzano-Narrows Bridge in New York City was built in 1964. The Golden Gate Bridge's clearance above high water averages 220 feet (67 m), and its towers rise to 746 feet (227 m) above the water.
    """
    
    text_to_check = """
    The Golden Gate Bridge is located in San Francisco and was completed in 1938. It has a main span of 4,200 feet and is painted bright red. The bridge is considered one of the most beautiful bridges in the world and is a major tourist attraction.
    """
    
    accuracy_result = client.evaluate_factual_accuracy(
        text_to_check,
        reference_info
    )
    print(f"Result: {accuracy_result['result']['judgment']}")
    
    # Example 3: Hallucination detection
    print("\n=== Example 3: Hallucination Detection ===")
    source_info = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France. The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people ascended it in 2015. The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
    """
    
    generated_text = """
    The Eiffel Tower, designed by Gustave Eiffel and completed in 1889, stands as Paris's most iconic landmark. At 330 meters tall, it was the world's tallest building until 1930 when the Empire State Building surpassed it. Made of approximately 18,000 iron pieces and 2.5 million rivets, the tower was almost demolished in 1909 but was saved because of its value as a radio transmission tower. During World War II, when Hitler visited Paris, the elevator cables were cut so he would have to climb the stairs if he wanted to reach the top. Today, the tower is repainted every seven years, requiring 60 tons of paint.
    """
    
    hallucination_result = client.detect_hallucinations(
        generated_text,
        source_info
    )
    print(f"Result: {hallucination_result['result']['judgment']}")
    
    # Example 4: Evaluate conversational AI response
    print("\n=== Example 4: Evaluate Conversational AI Response ===")
    user_query = "Can you explain quantum computing in simple terms?"
    ai_response = "Quantum computing is like having a super-powerful calculator that can try many answers at once instead of one at a time. It uses special particles that can exist in multiple states simultaneously, which allows it to solve certain problems much faster than regular computers. Think of it as being able to check all paths through a maze at the same time, rather than trying one path after another."
    
    response_eval_result = client.evaluate_conversational_ai_response(
        user_query,
        ai_response
    )
    print(f"Result: {response_eval_result['result']['judgment']}")
    
    # Example 5: Compare summarizations
    print("\n=== Example 5: Compare Summarizations ===")
    original_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    
    AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).
    
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    """
    
    summary_A = "AI is machine intelligence that perceives its environment and makes decisions to achieve goals. It includes applications like search engines, recommendation systems, voice assistants, self-driving cars, and game-playing systems. As AI advances, technologies once considered AI are often reclassified as routine."
    
    summary_B = "Artificial intelligence refers to computer systems that can perform tasks requiring human-like intelligence. These systems use algorithms to learn from data, recognize patterns, and make decisions. AI has applications in various fields including healthcare, finance, transportation, and entertainment. The field continues to evolve rapidly with advances in machine learning and neural networks."
    
    compare_result = client.compare_summarizations(
        original_text,
        summary_A,
        summary_B
    )
    print(f"Result: {compare_result['result']['judgment']}")
    
    # Example 6: Create a custom template
    print("\n=== Example 6: Create Custom Template ===")
    template_result = client.create_custom_template(
        template_name="Code Quality Evaluation",
        system_message="You are an expert software developer. Your task is to evaluate the quality of the provided code.",
        user_instruction_prefix="Please evaluate the following code for quality, readability, efficiency, and best practices:\n\n{evaluation_criteria}\n\nCode to evaluate:\n\n",
        user_instruction_suffix="\n\n{output_format_instruction}",
        output_parser_rules={
            "type": "json",
            "format": {
                "quality_score": "number (1-5)",
                "readability": "number (1-5)",
                "efficiency": "number (1-5)",
                "adheres_to_best_practices": "boolean",
                "suggestions": "array of strings"
            }
        },
        description="Template for evaluating code quality"
    )
    print(f"Created template: {template_result['template_id']} - {template_result['template_name']}")