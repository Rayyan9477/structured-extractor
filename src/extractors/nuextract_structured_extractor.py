"""
Structured Data Extractor using NuExtract 8B

Uses the numind/NuExtract-2.0-8B model to extract structured information from
document text with custom templates for different document types.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
import torch

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForVision2Seq,
    GenerationConfig
)
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import StructuredData, FieldConfidence


class NuExtractStructuredExtractor:
    """
    Structured data extraction using numind/NuExtract-2.0-8B.
    Extracts structured information based on configurable templates.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the NuExtract structured extractor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get model configuration
        self.extraction_config = config.get("extraction", {}).get("nuextract", {})
        self.model_name = self.extraction_config.get("model_name", "numind/NuExtract-2.0-8B")
        self.max_length = self.extraction_config.get("max_length", 4096)
        self.temperature = self.extraction_config.get("temperature", 0.1)
        self.top_p = self.extraction_config.get("top_p", 0.9)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.models_loaded = False
        
        # Templates for different document types
        self.templates = self._initialize_templates()
        
        self.logger.info(f"NuExtract Structured Extractor initialized with model: {self.model_name}")
    
    def _get_device(self) -> str:
        """Get optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize extraction templates from configuration.
        
        Returns:
            Dictionary of templates
        """
        templates = self.extraction_config.get("templates", {})
        
        # Add default template if none exist
        if not templates:
            templates = {
                "default": {
                    "name": "default",
                    "description": "Default structured data extraction template",
                    "schema": {
                        "title": "str",
                        "date": "str",
                        "sender": "str",
                        "receiver": "str",
                        "content": "str",
                        "key_points": "list[str]",
                        "metadata": "dict"
                    },
                    "examples": []
                },
                "medical_superbill": {
                    "name": "medical_superbill",
                    "description": "Medical superbill and healthcare document extraction template",
                    "schema": {
                        "patient_name": "str",
                        "patient_dob": "str",
                        "patient_id": "str",
                        "insurance_id": "str",
                        "service_date": "str",
                        "provider_name": "str",
                        "facility_name": "str",
                        "cpt_codes": "list[dict]",
                        "icd10_codes": "list[dict]",
                        "total_charges": "float",
                        "insurance_paid": "float",
                        "patient_paid": "float",
                        "balance_due": "float",
                        "diagnosis": "str",
                        "procedures": "list[str]",
                        "billing_address": "str",
                        "provider_npi": "str",
                        "tax_id": "str"
                    },
                    "examples": []
                },
                "patient_demographics": {
                    "name": "patient_demographics",
                    "description": "Patient demographic information extraction",
                    "schema": {
                        "first_name": "str",
                        "last_name": "str",
                        "date_of_birth": "str",
                        "gender": "str",
                        "address": "str",
                        "phone": "str",
                        "email": "str",
                        "emergency_contact": "str"
                    },
                    "examples": []
                }
            }
        
        self.logger.info(f"Initialized {len(templates)} extraction templates")
        return templates
    
    async def load_model(self) -> None:
        """Load the NuExtract model and tokenizer."""
        if self.models_loaded:
            return
            
        try:
            self.logger.info(f"Loading NuExtract model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Try different model classes based on the model type
            try:
                # First try with AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            except Exception as e1:
                self.logger.warning(f"Couldn't load with AutoModelForCausalLM: {e1}")
                try:
                    # Try with AutoModelForVision2Seq if it's a vision-language model
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    )
                except Exception as e2:
                    self.logger.warning(f"Couldn't load with AutoModelForVision2Seq: {e2}")
                    # Last resort: use AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    )
            
            # Skip moving model if using device_map="auto" (model is already distributed)
            # Only move if device_map was not used
            try:
                if hasattr(self.model, "device") and not hasattr(self.model, "hf_device_map"):
                    self.model.to(self.device)
            except RuntimeError as e:
                if "offloaded" in str(e):
                    self.logger.info("Model uses device mapping, skipping manual device placement")
                else:
                    raise
            
            self.models_loaded = True
            self.logger.info(f"NuExtract model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load NuExtract model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load NuExtract model: {e}")
    
    async def extract_structured_data(
        self,
        text: str,
        template_name: str = "default",
        custom_schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> StructuredData:
        """
        Extract structured data from text based on template.
        
        Args:
            text: The text to extract from
            template_name: The name of the template to use
            custom_schema: Optional custom schema to override template
            max_retries: Maximum number of extraction attempts
            
        Returns:
            Structured data extracted from text
        """
        if not self.models_loaded:
            await self.load_model()
        
        # Get template
        template = self.templates.get(template_name)
        if not template:
            self.logger.warning(f"Template '{template_name}' not found, using default")
            template = self.templates.get("default", {
                "schema": {
                    "content": "str",
                    "metadata": "dict"
                }
            })
        
        # Use custom schema if provided
        schema = custom_schema if custom_schema else template.get("schema", {})
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(text, schema, template)
        
        # Try extraction with retries
        for attempt in range(max_retries):
            try:
                # Generate extraction
                result_text = await self._generate_extraction(prompt)
                
                # Parse result
                extracted_data = self._parse_extraction_result(result_text, schema)
                
                # Validate and return
                validated_data = self._validate_extracted_data(extracted_data, schema)
                confidence = self._calculate_confidence(validated_data, schema)
                
                return StructuredData(
                    data=validated_data,
                    confidence=confidence,
                    model_name=self.model_name
                )
                
            except Exception as e:
                self.logger.warning(f"Extraction attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("All extraction attempts failed", exc_info=True)
                    return StructuredData(
                        data={"error": str(e)},
                        confidence=0.0,
                        model_name=self.model_name
                    )
    
    def _create_extraction_prompt(
        self,
        text: str,
        schema: Dict[str, Any],
        template: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an extraction prompt based on the template and text.
        
        Args:
            text: The text to extract from
            schema: The schema to extract
            template: Optional template with examples
            
        Returns:
            Formatted extraction prompt
        """
        # Format schema as string
        schema_str = json.dumps(schema, indent=2)
        
        # Get examples from template
        examples = []
        if template and "examples" in template:
            examples = template.get("examples", [])
        
        # Check if model is instruction-tuned
        is_instruction_tuned = "instruct" in self.model_name.lower()
        
        # Create prompt based on model type
        if is_instruction_tuned:
            # For instruction-tuned models like Mixtral-Instruct
            prompt = f"""<s>[INST] You are an expert data extraction assistant. Extract structured information from the text according to this JSON schema:
{schema_str}

Here is the text to extract information from:
{text}

Return ONLY the extracted data as valid JSON. Do not include any explanations, notes, or comments.
Format your response as a valid, parsable JSON object. [/INST]"""
        else:
            # For foundation models like Phi-2
            prompt = f"""You are an expert data extraction assistant. Extract structured information from the text according to this JSON schema:
{schema_str}

Here is the text to extract information from:
{text}

Return ONLY the extracted data as valid JSON. Do not include any explanations, notes, or comments.
Format your response as a valid, parsable JSON object.
"""
        
        # Add examples if available
        if examples:
            examples_text = "\n\nHere are some examples to guide your extraction:"
            for i, example in enumerate(examples):
                if "text" in example and "extracted" in example:
                    examples_text += f"\n\nExample {i+1}:\nText: {example['text']}\nExtracted: {json.dumps(example['extracted'], indent=2)}"
            
            if is_instruction_tuned:
                prompt = f"""<s>[INST] You are an expert data extraction assistant. Extract structured information from the text according to this JSON schema:
{schema_str}
{examples_text}

Here is the text to extract information from:
{text}

Return ONLY the extracted data as valid JSON. Do not include any explanations, notes, or comments.
Format your response as a valid, parsable JSON object. [/INST]"""
            else:
                prompt = f"""You are an expert data extraction assistant. Extract structured information from the text according to this JSON schema:
{schema_str}
{examples_text}

Here is the text to extract information from:
{text}

Return ONLY the extracted data as valid JSON. Do not include any explanations, notes, or comments.
Format your response as a valid, parsable JSON object.
"""
        
        return prompt
    
    async def _generate_extraction(self, prompt: str) -> str:
        """
        Generate structured extraction from the prompt.
        
        Args:
            prompt: The extraction prompt
            
        Returns:
            Raw extraction result text
        """
        try:
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Create generation config
            generation_config = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Generate output
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            
            # Decode output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            # Different handling for different model types
            is_instruction_tuned = "[/INST]" in prompt
            
            if is_instruction_tuned:
                # For instruction-tuned models like Mixtral
                parts = decoded_output.split("[/INST]")
                if len(parts) > 1:
                    result_text = parts[1].strip()
                else:
                    # Fallback if splitting by [/INST] didn't work
                    result_text = decoded_output.strip()
            else:
                # For foundation models without instruction format
                # Try to find where the model's response begins
                # First check if the decoded output contains the entire prompt
                if decoded_output.startswith(prompt):
                    result_text = decoded_output[len(prompt):].strip()
                else:
                    # Otherwise, try to find where the JSON output begins
                    result_text = decoded_output.strip()
            
            # Find the JSON part in the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                result_text = result_text[json_start:json_end+1].strip()
            
            return result_text
            
        except Exception as e:
            self.logger.error(f"Error generating extraction: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate extraction: {e}")
    
    def _parse_extraction_result(self, result_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the extraction result into a structured data dictionary.
        
        Args:
            result_text: The raw extraction result
            schema: The schema for validation
            
        Returns:
            Parsed extraction data
        """
        try:
            # Find JSON in the response
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}")
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = result_text[start_idx:end_idx+1]
                # Fix common JSON issues
                json_text = self._fix_json_issues(json_text)
                # Parse JSON
                data = json.loads(json_text)
                return data
            else:
                self.logger.warning("No JSON object found in extraction result")
                return {"raw_extraction": result_text}
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse extraction result as JSON: {e}")
            # Try to recover by parsing as key-value pairs
            return {"error": "JSON parsing failed", "raw_extraction": result_text}
    
    def _fix_json_issues(self, json_text: str) -> str:
        """
        Fix common JSON syntax issues in extraction results.
        
        Args:
            json_text: Raw JSON text to fix
            
        Returns:
            Fixed JSON text
        """
        import re
        
        # Simple approach - fix the most common issues without complex regex
        # Fix trailing commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix missing quotes around keys (simple pattern)
        json_text = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_text)
        
        # Fix unescaped quotes (simple replacement)
        # First pass: protect already escaped quotes
        json_text = json_text.replace('\\"', '__ESCAPED_QUOTE__')
        # Second pass: escape unescaped quotes in values
        lines = json_text.split('\n')
        fixed_lines = []
        for line in lines:
            if ':' in line and '"' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1]
                    # Simple escaping for values
                    if value_part.strip().startswith('"') and value_part.strip().endswith('"'):
                        value_content = value_part.strip()[1:-1]
                        value_content = value_content.replace('"', '\\"')
                        value_part = value_part.replace(value_part.strip()[1:-1], value_content)
                    line = key_part + ':' + value_part
            fixed_lines.append(line)
        json_text = '\n'.join(fixed_lines)
        
        # Restore escaped quotes
        json_text = json_text.replace('__ESCAPED_QUOTE__', '\\"')
        
        return json_text
    
    def _validate_extracted_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean up extracted data against the schema.
        
        Args:
            data: The extracted data
            schema: The schema to validate against
            
        Returns:
            Validated data
        """
        validated = {}
        
        for key, expected_type in schema.items():
            if key in data:
                # Store the value with basic type validation
                value = data[key]
                
                # Ensure lists for list fields
                if expected_type.startswith("list") and not isinstance(value, list):
                    if isinstance(value, str):
                        # Try to convert comma-separated string to list
                        validated[key] = [item.strip() for item in value.split(",")]
                    else:
                        # Convert single value to list
                        validated[key] = [value]
                else:
                    validated[key] = value
            else:
                # Field is missing
                if expected_type.startswith("list"):
                    validated[key] = []
                elif expected_type == "dict":
                    validated[key] = {}
                else:
                    validated[key] = None
        
        return validated
    
    def _calculate_confidence(self, data: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the extracted data.
        
        Args:
            data: The extracted data
            schema: The schema used for extraction
            
        Returns:
            Confidence score between 0 and 1
        """
        # Count non-empty fields
        non_empty = 0
        total = len(schema)
        
        for key in schema:
            value = data.get(key)
            if value is not None:
                if isinstance(value, list) and len(value) > 0:
                    non_empty += 1
                elif isinstance(value, dict) and len(value) > 0:
                    non_empty += 1
                elif isinstance(value, str) and value.strip():
                    non_empty += 1
                elif not isinstance(value, (str, list, dict)) and value:
                    non_empty += 1
        
        # Calculate confidence based on field coverage
        confidence = non_empty / total if total > 0 else 0.0
        
        # Apply scaling factor (more fields should give higher confidence)
        scaling = min(1.0, (total / 10) * 0.5 + 0.5)
        
        return confidence * scaling 