"""
NuExtract Integration for Medical Document Understanding

Integrates numind/NuExtract-2.0-8B for structured information extraction from OCR text,
with specialized templates for medical superbill formats.
"""

import json
import asyncio
import torch
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, GenerationConfig
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import (
    PatientData, CPTCode, ICD10Code, ServiceInfo, 
    ProviderInfo, FinancialInfo, FieldConfidence
)


@dataclass
class ExtractionTemplate:
    """Template for structured extraction."""
    name: str
    description: str
    schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    confidence_threshold: float = 0.7


class NuExtractEngine:
    """NuExtract-based structured information extraction engine."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize NuExtract engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get model configuration
        self.model_config = config.get_model_config("extraction", "nuextract")
        self.model_name = "numind/NuExtract-2.0-8B"  # Using the 8B model
        self.max_length = self.model_config.get("max_length", 4096)  # Increased for 8B model
        self.temperature = self.model_config.get("temperature", 0.1)
        self.top_p = self.model_config.get("top_p", 0.9)
        
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = self._get_device()
        
        # Initialize extraction templates
        self.templates = self._initialize_templates()
        
        self.logger.info(f"Initializing NuExtract with model: {self.model_name}")
    
    def _get_device(self) -> str:
        """Get optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_model(self) -> None:
        """Load the NuExtract model and tokenizer."""
        try:
            self.logger.info("Loading NuExtract model...")
            
            # Get correct model path
            model_path = Path(self.config.get_model_path(self.model_name))
            
            if not model_path.exists():
                self.logger.warning(f"Local model not found at {model_path}, using remote model")
                model_path = self.model_name
            
            # Load processor and model for Qwen2.5-VL based NuExtract
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.processor = AutoProcessor.from_pretrained(
                str(model_path), 
                trust_remote_code=True,
                padding_side='left',
                use_fast=True
            )
            
            # Load the model for Vision2Seq tasks with better error handling
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as model_error:
                self.logger.warning(f"Failed to load as Vision2Seq model: {model_error}")
                # Fallback to CausalLM for text-only processing
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            self.model.eval()
            
            self.logger.info(f"NuExtract model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load NuExtract model: {e}")
            raise
    
    def _initialize_templates(self) -> Dict[str, ExtractionTemplate]:
        """Initialize extraction templates for different document types."""
        templates = {}
        
        # Medical superbill template
        templates["medical_superbill"] = ExtractionTemplate(
            name="medical_superbill",
            description="Extract structured data from medical superbill documents",
            schema={
                "patients": [
                    {
                        "patient_info": {
                            "first_name": "string",
                            "last_name": "string",
                            "middle_name": "string",
                            "date_of_birth": "date",
                            "patient_id": "string",
                            "address": {
                                "street": "string",
                                "city": "string", 
                                "state": "string",
                                "zip_code": "string"
                            },
                            "phone": "string",
                            "email": "string"
                        },
                        "insurance_info": {
                            "insurance_company": "string",
                            "policy_number": "string",
                            "group_number": "string",
                            "subscriber_id": "string"
                        },
                        "medical_codes": {
                            "cpt_codes": [
                                {
                                    "code": "string",
                                    "description": "string",
                                    "modifier": "string",
                                    "units": "number",
                                    "charge": "number"
                                }
                            ],
                            "icd10_codes": [
                                {
                                    "code": "string",
                                    "description": "string",
                                    "is_primary": "boolean"
                                }
                            ]
                        },
                        "service_info": {
                            "date_of_service": "date",
                            "claim_date": "date",
                            "place_of_service": "string",
                            "visit_type": "string",
                            "chief_complaint": "string"
                        },
                        "financial_info": {
                            "total_charges": "number",
                            "amount_paid": "number",
                            "outstanding_balance": "number",
                            "copay": "number",
                            "deductible": "number"
                        }
                    }
                ],
                "provider_info": {
                    "provider_name": "string",
                    "npi_number": "string",
                    "practice_name": "string",
                    "address": {
                        "street": "string",
                        "city": "string",
                        "state": "string", 
                        "zip_code": "string"
                    },
                    "phone": "string",
                    "tax_id": "string"
                }
            },
            examples=[
                {
                    "input": "Patient: John Doe DOB: 01/15/1980 Date of Service: 03/15/2024 CPT: 99213 Office Visit $150.00 Diagnosis: M54.5 Low back pain",
                    "output": {
                        "patients": [
                            {
                                "patient_info": {
                                    "first_name": "John",
                                    "last_name": "Doe",
                                    "date_of_birth": "1980-01-15"
                                },
                                "medical_codes": {
                                    "cpt_codes": [
                                        {
                                            "code": "99213",
                                            "description": "Office Visit",
                                            "charge": 150.00
                                        }
                                    ],
                                    "icd10_codes": [
                                        {
                                            "code": "M54.5",
                                            "description": "Low back pain",
                                            "is_primary": True
                                        }
                                    ]
                                },
                                "service_info": {
                                    "date_of_service": "2024-03-15"
                                },
                                "financial_info": {
                                    "total_charges": 150.00
                                }
                            }
                        ]
                    }
                }
            ]
        )
        
        # Patient demographics template
        templates["patient_demographics"] = ExtractionTemplate(
            name="patient_demographics",
            description="Extract patient demographic information",
            schema={
                "patient_name": "string",
                "date_of_birth": "date",
                "address": "string",
                "phone": "string",
                "email": "string",
                "insurance_info": "string"
            },
            examples=[]
        )
        
        # Medical codes template
        templates["medical_codes"] = ExtractionTemplate(
            name="medical_codes",
            description="Extract CPT and ICD-10 codes with descriptions",
            schema={
                "cpt_codes": [{"code": "string", "description": "string", "charge": "number"}],
                "icd10_codes": [{"code": "string", "description": "string"}]
            },
            examples=[]
        )
        
        return templates
    
    async def extract_structured_data(
        self, 
        text: str, 
        template_name: str = "medical_superbill",
        custom_schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using specified template.
        
        Args:
            text: Input text to extract from
            template_name: Name of extraction template to use
            custom_schema: Custom schema to override template
            max_retries: Maximum number of retries on failure
            
        Returns:
            Extracted structured data
        """
        if self.model is None:
            await self.load_model()
        
        self.logger.debug(f"Extracting structured data using template: {template_name}")
        
        retries = 0
        while retries <= max_retries:
            try:
                # Get template
                template = self.templates.get(template_name)
                if template is None and custom_schema is None:
                    raise ValueError(f"Unknown template: {template_name}")
                
                # Use custom schema if provided
                schema = custom_schema if custom_schema else template.schema
                
                # Create extraction prompt
                prompt = self._create_extraction_prompt(text, schema, template)
                
                # Generate structured output
                result = await self._generate_extraction(prompt)
                
                # Parse and validate result
                parsed_result = self._parse_extraction_result(result, schema)
                
                if not parsed_result:
                    raise ValueError("Failed to parse extraction result")
                
                self.logger.debug("Structured data extraction completed successfully")
                return parsed_result
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    self.logger.error(f"Structured data extraction failed after {max_retries} retries: {e}")
                    return {}
                
                self.logger.warning(f"Extraction attempt {retries} failed: {e}. Retrying...")
                await asyncio.sleep(1)  # Wait before retry
    
    def _create_extraction_prompt(
        self, 
        text: str, 
        schema: Dict[str, Any], 
        template: Optional[ExtractionTemplate] = None
    ) -> str:
        """Create prompt for structured extraction."""
        
        # Format schema for prompt
        schema_str = json.dumps(schema, indent=2)
        
        # Create instruction prompt
        prompt = f"""You are an expert medical data extraction system. Extract structured information from the given medical document text according to the specified JSON schema.

Schema to follow:
{schema_str}

Instructions:
1. Extract all relevant information that matches the schema
2. If information is not found, omit the field or use null
3. Ensure all dates are in YYYY-MM-DD format
4. Ensure all medical codes follow proper formatting (CPT: 5 digits, ICD-10: letter + 2 digits + optional decimal)
5. Extract monetary amounts as numbers without currency symbols
6. Be precise and only extract information that is clearly present in the text

Text to extract from:
{text}

Extracted JSON:"""
        
        return prompt
    
    async def _generate_extraction(self, prompt: str) -> str:
        """Generate extraction using the model."""
        try:
            # Prepare messages for Qwen2.5-VL based NuExtract
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs (text-only for structured extraction)
            inputs = self.processor(
                text=[text],
                images=None,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = inputs.to(self.model.device)
            
            # Generate with optimized parameters for 8B model
            generation_config = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 2048,
                "temperature": self.temperature if hasattr(self, 'temperature') else 0.1,
                "top_p": self.top_p if hasattr(self, 'top_p') else 0.9,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode output - extract only the generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return "{}"
    
    def _parse_extraction_result(self, result_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate extraction result."""
        try:
            # Try to find JSON in the result
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = result_text[json_start:json_end]
                parsed = json.loads(json_text)
                return self._validate_extracted_data(parsed, schema)
            else:
                self.logger.warning("No valid JSON found in extraction result")
                return {}
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}")
            # Try to fix common JSON issues
            fixed_result = self._fix_json_issues(result_text)
            try:
                parsed = json.loads(fixed_result)
                return self._validate_extracted_data(parsed, schema)
            except:
                return {}
        except Exception as e:
            self.logger.error(f"Result parsing failed: {e}")
            return {}
    
    def _fix_json_issues(self, json_text: str) -> str:
        """Try to fix common JSON formatting issues."""
        if not json_text:
            return "{}"
            
        # Remove any text before the first {
        json_start = json_text.find('{')
        if json_start > 0:
            json_text = json_text[json_start:]
        elif json_start < 0:
            # No JSON object found
            return "{}"
        
        # Remove any text after the last }
        json_end = json_text.rfind('}')
        if json_end >= 0:
            json_text = json_text[:json_end + 1]
        else:
            # No closing bracket
            return "{}"
        
        # Fix common issues
        json_text = json_text.replace("'", '"')  # Single quotes to double quotes
        json_text = json_text.replace('True', 'true')  # Python bool to JSON bool
        json_text = json_text.replace('False', 'false')
        json_text = json_text.replace('None', 'null')
        
        # Fix trailing commas (common JSON parsing error)
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix missing quotes around keys
        json_text = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_text)
        
        # Ensure consistent spacing
        json_text = re.sub(r'"\s*:\s*"', '": "', json_text)
        json_text = re.sub(r'"\s*:\s*\{', '": {', json_text)
        json_text = re.sub(r'"\s*:\s*\[', '": [', json_text)
        
        # Try to parse and regenerate to fix subtle issues
        try:
            parsed = json.loads(json_text)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            self.logger.warning("Could not fully recover JSON, returning partially fixed version")
            return json_text
    
    def _validate_extracted_data(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against schema."""
        # Basic validation - ensure required fields exist
        # This is a simplified validation; in production, you might want more robust validation
        
        validated_data = {}
        
        for key, value in data.items():
            if key in schema:
                validated_data[key] = value
        
        return validated_data
    
    async def extract_from_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text and structure from an image using NuExtract.
        
        Args:
            image_bytes: Image bytes to process
            
        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            if self.model is None:
                await self.load_model()
                
            from PIL import Image
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Prepare the prompt for text extraction
            prompt = "Extract all text from this document, including tables and structured data. Return the raw text with line breaks preserved."
            
            # Prepare messages for the Vision2Seq model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = inputs.to(self.model.device)
            
            # Generate output with optimized parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=2048,
                    temperature=0.1,
                    top_p=0.9,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode output - extract only the generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            extracted_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Clean up the extracted text
            extracted_text = extracted_text.strip()
            if extracted_text:
                extracted_text = '\n'.join(line.strip() for line in extracted_text.split('\n') if line.strip())
            
            # Create OCRResult-like structure for compatibility
            result = type('OCRResult', (), {
                'text': extracted_text,
                'confidence': 0.85 if extracted_text else 0.0
            })()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in NuExtract image extraction: {str(e)}")
            # Return OCRResult-like structure for compatibility
            result = type('OCRResult', (), {
                'text': '',
                'confidence': 0.0
            })()
            return result
    
    async def extract_patient_data(self, text: str, patient_index: int = 0) -> Optional[PatientData]:
        """
        Extract patient data from text using NuExtract.
        
        Args:
            text: Input text to extract from
            patient_index: Index of the patient in multi-patient documents
            
        Returns:
            Extracted patient data or None if extraction fails
        """
        try:
            # Extract structured data
            extracted = await self.extract_structured_data(text, "medical_superbill")
            
            if not extracted or "patients" not in extracted:
                return None
            
            patients = extracted["patients"]
            if len(patients) <= patient_index:
                return None
            
            patient_data = patients[patient_index]
            
            # Convert to PatientData model
            return self._convert_to_patient_data(patient_data, extracted.get("provider_info"))
            
        except Exception as e:
            self.logger.error(f"Patient data extraction failed: {e}")
            return None
    
    def _convert_to_patient_data(
        self, 
        patient_data: Dict[str, Any], 
        provider_data: Optional[Dict[str, Any]] = None
    ) -> PatientData:
        """Convert extracted data to PatientData model."""
        from src.core.data_schema import Address, ContactInfo, InsuranceInfo, ServiceInfo, ProviderInfo, FinancialInfo
        
        # Patient demographics
        patient_info = patient_data.get("patient_info", {})
        
        # Address
        address_data = patient_info.get("address", {})
        address = Address(
            street=address_data.get("street"),
            city=address_data.get("city"),
            state=address_data.get("state"),
            zip_code=address_data.get("zip_code")
        ) if any(address_data.values()) else None
        
        # Contact info
        contact = ContactInfo(
            phone=patient_info.get("phone"),
            email=patient_info.get("email")
        ) if patient_info.get("phone") or patient_info.get("email") else None
        
        # Insurance
        insurance_data = patient_data.get("insurance_info", {})
        insurance = InsuranceInfo(
            insurance_company=insurance_data.get("insurance_company"),
            policy_number=insurance_data.get("policy_number"),
            group_number=insurance_data.get("group_number"),
            subscriber_id=insurance_data.get("subscriber_id")
        ) if any(insurance_data.values()) else None
        
        # CPT codes
        cpt_codes = []
        for cpt_data in patient_data.get("medical_codes", {}).get("cpt_codes", []):
            cpt_codes.append(CPTCode(
                code=cpt_data.get("code", ""),
                description=cpt_data.get("description"),
                modifier=cpt_data.get("modifier"),
                units=cpt_data.get("units"),
                charge=cpt_data.get("charge")
            ))
        
        # ICD-10 codes
        icd10_codes = []
        for icd_data in patient_data.get("medical_codes", {}).get("icd10_codes", []):
            icd10_codes.append(ICD10Code(
                code=icd_data.get("code", ""),
                description=icd_data.get("description"),
                is_primary=icd_data.get("is_primary", False)
            ))
        
        # Service info
        service_data = patient_data.get("service_info", {})
        service_info = ServiceInfo(
            date_of_service=self._parse_date(service_data.get("date_of_service")),
            claim_date=self._parse_date(service_data.get("claim_date")),
            place_of_service=service_data.get("place_of_service"),
            visit_type=service_data.get("visit_type"),
            chief_complaint=service_data.get("chief_complaint")
        ) if any(service_data.values()) else None
        
        # Financial info
        financial_data = patient_data.get("financial_info", {})
        financial = FinancialInfo(
            total_charges=financial_data.get("total_charges"),
            amount_paid=financial_data.get("amount_paid"),
            outstanding_balance=financial_data.get("outstanding_balance"),
            copay=financial_data.get("copay"),
            deductible=financial_data.get("deductible")
        ) if any(v is not None for v in financial_data.values()) else None
        
        # Provider info (if available)
        provider = None
        if provider_data:
            provider_address_data = provider_data.get("address", {})
            provider_address = Address(
                street=provider_address_data.get("street"),
                city=provider_address_data.get("city"),
                state=provider_address_data.get("state"),
                zip_code=provider_address_data.get("zip_code")
            ) if any(provider_address_data.values()) else None
            
            provider_contact = ContactInfo(
                phone=provider_data.get("phone")
            ) if provider_data.get("phone") else None
            
            provider = ProviderInfo(
                name=provider_data.get("provider_name"),
                npi_number=provider_data.get("npi_number"),
                practice_name=provider_data.get("practice_name"),
                address=provider_address,
                contact=provider_contact,
                tax_id=provider_data.get("tax_id")
            )
        
        return PatientData(
            first_name=patient_info.get("first_name"),
            last_name=patient_info.get("last_name"),
            middle_name=patient_info.get("middle_name"),
            date_of_birth=self._parse_date(patient_info.get("date_of_birth")),
            patient_id=patient_info.get("patient_id"),
            address=address,
            contact=contact,
            phone=patient_info.get("phone"),
            insurance=insurance,
            cpt_codes=cpt_codes,
            icd10_codes=icd10_codes,
            service_info=service_info,
            financial_info=financial,
            provider=provider,
            extraction_confidence=0.8  # Default confidence
        )
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None
        
        try:
            import dateparser
            from datetime import datetime
            
            # First try standard formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.date()
                except ValueError:
                    continue
            
            # Fallback to dateparser
            parsed = dateparser.parse(date_str)
            return parsed.date() if parsed else None
        except Exception as e:
            self.logger.debug(f"Date parsing failed for '{date_str}': {e}")
            return None
