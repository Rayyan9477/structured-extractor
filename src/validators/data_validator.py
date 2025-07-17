"""
Data Validators for Medical Superbill Extraction

Implements comprehensive validation for CPT codes, ICD-10 codes, dates,
PHI anonymization, and other medical data validation requirements.
"""

import re
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import date, datetime
from pathlib import Path
import json

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import PatientData, CPTCode, ICD10Code
from .date_validator import DateValidator


class CPTCodeValidator:
    """Validates CPT codes against official databases and format rules."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize CPT code validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load CPT code database if available
        self.cpt_database = self._load_cpt_database()
        
        # CPT code ranges and categories
        self.cpt_ranges = {
            'evaluation_management': (99202, 99499),
            'anesthesia': (100, 1999),
            'surgery': (10021, 69990),
            'radiology': (70010, 79999),
            'pathology_lab': (80047, 89398),
            'medicine': (90281, 99607)
        }
    
    def _load_cpt_database(self) -> Dict[str, Dict[str, Any]]:
        """Load CPT code database from configuration or file."""
        try:
            # Try to load from configuration
            cpt_db_path = self.config.get("validation", {}).get("cpt_database_path")
            
            if cpt_db_path and Path(cpt_db_path).exists():
                with open(cpt_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Return minimal built-in database
            return self._get_common_cpt_codes()
            
        except Exception as e:
            self.logger.warning(f"Could not load CPT database: {e}")
            return self._get_common_cpt_codes()
    
    def _get_common_cpt_codes(self) -> Dict[str, Dict[str, Any]]:
        """Return common CPT codes for validation."""
        return {
            "99213": {
                "description": "Office visit, established patient, low complexity",
                "category": "evaluation_management",
                "active": True
            },
            "99214": {
                "description": "Office visit, established patient, moderate complexity",
                "category": "evaluation_management", 
                "active": True
            },
            "99215": {
                "description": "Office visit, established patient, high complexity",
                "category": "evaluation_management",
                "active": True
            },
            "99202": {
                "description": "Office visit, new patient, straightforward",
                "category": "evaluation_management",
                "active": True
            },
            "99203": {
                "description": "Office visit, new patient, low complexity",
                "category": "evaluation_management",
                "active": True
            },
            "99204": {
                "description": "Office visit, new patient, moderate complexity", 
                "category": "evaluation_management",
                "active": True
            },
            "99205": {
                "description": "Office visit, new patient, high complexity",
                "category": "evaluation_management",
                "active": True
            },
            "90834": {
                "description": "Psychotherapy, 45 minutes",
                "category": "medicine",
                "active": True
            },
            "90837": {
                "description": "Psychotherapy, 60 minutes", 
                "category": "medicine",
                "active": True
            },
            "93000": {
                "description": "Electrocardiogram, complete",
                "category": "medicine",
                "active": True
            }
        }
    
    def validate_cpt_code(self, cpt_code: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a CPT code.
        
        Args:
            cpt_code: CPT code to validate
            
        Returns:
            Tuple of (is_valid, validation_message, code_info)
        """
        # Basic format validation
        if not re.match(r'^\d{5}$', cpt_code):
            return False, "Invalid CPT code format (must be 5 digits)", {}
        
        # Range validation
        code_num = int(cpt_code)
        if code_num < 100 or code_num > 99999:
            return False, "CPT code out of valid range", {}
        
        # Database validation
        if cpt_code in self.cpt_database:
            code_info = self.cpt_database[cpt_code]
            
            if not code_info.get("active", True):
                return False, "CPT code is inactive/deprecated", code_info
            
            return True, "Valid CPT code", code_info
        
        # Category validation based on ranges
        category = self._get_cpt_category(code_num)
        if category:
            return True, f"Valid CPT code format (category: {category})", {
                "category": category,
                "description": "Description not available",
                "active": True
            }
        
        return False, "Unknown CPT code", {}
    
    def _get_cpt_category(self, code_num: int) -> Optional[str]:
        """Get CPT code category based on number ranges."""
        for category, (start, end) in self.cpt_ranges.items():
            if start <= code_num <= end:
                return category
        return None
    
    def validate_cpt_codes(self, cpt_codes: List[CPTCode]) -> List[Dict[str, Any]]:
        """
        Validate multiple CPT codes.
        
        Args:
            cpt_codes: List of CPT codes to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        for cpt in cpt_codes:
            is_valid, message, code_info = self.validate_cpt_code(cpt.code)
            
            results.append({
                'code': cpt.code,
                'is_valid': is_valid,
                'message': message,
                'code_info': code_info,
                'original_confidence': cpt.confidence
            })
        
        return results


class ICD10CodeValidator:
    """Validates ICD-10 codes against format rules and databases."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize ICD-10 code validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load ICD-10 database if available
        self.icd10_database = self._load_icd10_database()
    
    def _load_icd10_database(self) -> Dict[str, Dict[str, Any]]:
        """Load ICD-10 code database from configuration or file."""
        try:
            # Try to load from configuration
            icd10_db_path = self.config.get("validation", {}).get("icd10_database_path")
            
            if icd10_db_path and Path(icd10_db_path).exists():
                with open(icd10_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Return minimal built-in database
            return self._get_common_icd10_codes()
            
        except Exception as e:
            self.logger.warning(f"Could not load ICD-10 database: {e}")
            return self._get_common_icd10_codes()
    
    def _get_common_icd10_codes(self) -> Dict[str, Dict[str, Any]]:
        """Return common ICD-10 codes for validation."""
        return {
            "Z00.00": {
                "description": "Encounter for general adult medical examination without abnormal findings",
                "category": "Z00-Z13",
                "active": True
            },
            "Z12.31": {
                "description": "Encounter for screening mammogram for malignant neoplasm of breast",
                "category": "Z00-Z13", 
                "active": True
            },
            "I10": {
                "description": "Essential (primary) hypertension",
                "category": "I00-I99",
                "active": True
            },
            "E11.9": {
                "description": "Type 2 diabetes mellitus without complications",
                "category": "E00-E89",
                "active": True
            },
            "M79.3": {
                "description": "Panniculitis, unspecified",
                "category": "M00-M99",
                "active": True
            },
            "F41.9": {
                "description": "Anxiety disorder, unspecified",
                "category": "F01-F99",
                "active": True
            },
            "J44.1": {
                "description": "Chronic obstructive pulmonary disease with acute exacerbation",
                "category": "J00-J99",
                "active": True
            },
            "N39.0": {
                "description": "Urinary tract infection, site not specified",
                "category": "N00-N99",
                "active": True
            }
        }
    
    def validate_icd10_code(self, icd10_code: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate an ICD-10 code.
        
        Args:
            icd10_code: ICD-10 code to validate
            
        Returns:
            Tuple of (is_valid, validation_message, code_info)
        """
        # Remove any whitespace
        icd10_code = icd10_code.strip().upper()
        
        # Basic format validation
        # ICD-10 format: Letter + 2 digits + optional decimal + up to 3 more characters
        if not re.match(r'^[A-Z]\d{2}(\.[A-Z0-9]{1,3})?$', icd10_code):
            return False, "Invalid ICD-10 code format", {}
        
        # Database validation
        if icd10_code in self.icd10_database:
            code_info = self.icd10_database[icd10_code]
            
            if not code_info.get("active", True):
                return False, "ICD-10 code is inactive/deprecated", code_info
            
            return True, "Valid ICD-10 code", code_info
        
        # Basic category validation
        category = self._get_icd10_category(icd10_code[0])
        if category:
            return True, f"Valid ICD-10 code format (category: {category})", {
                "category": category,
                "description": "Description not available",
                "active": True
            }
        
        return False, "Unknown ICD-10 code format", {}
    
    def _get_icd10_category(self, first_letter: str) -> Optional[str]:
        """Get ICD-10 category based on first letter."""
        categories = {
            'A': 'Certain infectious and parasitic diseases',
            'B': 'Certain infectious and parasitic diseases',
            'C': 'Neoplasms', 
            'D': 'Diseases of the blood and blood-forming organs',
            'E': 'Endocrine, nutritional and metabolic diseases',
            'F': 'Mental, Behavioral and Neurodevelopmental disorders',
            'G': 'Diseases of the nervous system',
            'H': 'Diseases of the eye and ear',
            'I': 'Diseases of the circulatory system',
            'J': 'Diseases of the respiratory system',
            'K': 'Diseases of the digestive system',
            'L': 'Diseases of the skin and subcutaneous tissue',
            'M': 'Diseases of the musculoskeletal system',
            'N': 'Diseases of the genitourinary system',
            'O': 'Pregnancy, childbirth and the puerperium',
            'P': 'Certain conditions originating in the perinatal period',
            'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
            'R': 'Symptoms, signs and abnormal clinical and laboratory findings',
            'S': 'Injury, poisoning and certain other consequences of external causes',
            'T': 'Injury, poisoning and certain other consequences of external causes',
            'U': 'Codes for special purposes',
            'V': 'External causes of morbidity',
            'W': 'External causes of morbidity',
            'X': 'External causes of morbidity',
            'Y': 'External causes of morbidity',
            'Z': 'Factors influencing health status and contact with health services'
        }
        
        return categories.get(first_letter)
    
    def validate_icd10_codes(self, icd10_codes: List[ICD10Code]) -> List[Dict[str, Any]]:
        """
        Validate multiple ICD-10 codes.
        
        Args:
            icd10_codes: List of ICD-10 codes to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        for icd in icd10_codes:
            is_valid, message, code_info = self.validate_icd10_code(icd.code)
            
            results.append({
                'code': icd.code,
                'is_valid': is_valid,
                'message': message,
                'code_info': code_info,
                'original_confidence': icd.confidence
            })
        
        return results


class DateValidator:
    """Validates dates and date ranges in medical contexts."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize date validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Date validation configuration
        date_config = self.config.get("validation", {}).get("dates", {})
        self.min_birth_year = date_config.get("min_birth_year", 1900)
        self.max_future_days = date_config.get("max_future_days", 365)
    
    def validate_date_of_birth(self, dob: date) -> Tuple[bool, str]:
        """
        Validate date of birth.
        
        Args:
            dob: Date of birth to validate
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        today = date.today()
        
        # Check if date is in the future
        if dob > today:
            return False, "Date of birth cannot be in the future"
        
        # Check minimum year
        if dob.year < self.min_birth_year:
            return False, f"Date of birth year cannot be before {self.min_birth_year}"
        
        # Check if person would be unreasonably old
        age = (today - dob).days / 365.25
        if age > 150:
            return False, "Date of birth indicates unrealistic age (>150 years)"
        
        return True, "Valid date of birth"
    
    def validate_service_date(self, service_date: date, dob: Optional[date] = None) -> Tuple[bool, str]:
        """
        Validate service date.
        
        Args:
            service_date: Service date to validate
            dob: Optional date of birth for additional validation
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        today = date.today()
        
        # Check if service date is too far in the future
        future_limit = today.replace(year=today.year + 1)  # Allow up to 1 year in future
        if service_date > future_limit:
            return False, "Service date is too far in the future"
        
        # Check if service date is too far in the past
        past_limit = today.replace(year=today.year - 10)  # Allow up to 10 years in past
        if service_date < past_limit:
            return False, "Service date is too far in the past"
        
        # Cross-validate with date of birth if provided
        if dob:
            if service_date < dob:
                return False, "Service date cannot be before date of birth"
            
            # Check if service was provided to a minor (might need special handling)
            age_at_service = (service_date - dob).days / 365.25
            if age_at_service < 0:
                return False, "Invalid: service before birth"
        
        return True, "Valid service date"
    
    def validate_date_ranges(self, start_date: date, end_date: date) -> Tuple[bool, str]:
        """
        Validate date ranges.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if start_date > end_date:
            return False, "Start date cannot be after end date"
        
        # Check if range is unreasonably long
        range_days = (end_date - start_date).days
        if range_days > 365:  # More than a year
            return False, "Date range exceeds reasonable limits (>1 year)"
        
        return True, "Valid date range"


class PHIAnonymizer:
    """Anonymizes PHI (Protected Health Information) in medical data."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize PHI anonymizer.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Anonymization configuration
        self.phi_config = self.config.get("phi_anonymization", {})
        self.anonymize_names = self.phi_config.get("anonymize_names", True)
        self.anonymize_addresses = self.phi_config.get("anonymize_addresses", True)
        self.anonymize_phones = self.phi_config.get("anonymize_phones", True)
        self.anonymize_ssn = self.phi_config.get("anonymize_ssn", True)
        self.anonymize_dates = self.phi_config.get("anonymize_dates", False)
    
    def anonymize_patient_data(self, patient_data: PatientData) -> PatientData:
        """
        Anonymize PHI in patient data.
        
        Args:
            patient_data: Patient data to anonymize
            
        Returns:
            Anonymized patient data
        """
        # Create a copy to avoid modifying original
        anonymized = patient_data.model_copy(deep=True)
        
        # Anonymize names
        if self.anonymize_names:
            if anonymized.first_name:
                anonymized.first_name = self._anonymize_name(anonymized.first_name)
            if anonymized.last_name:
                anonymized.last_name = self._anonymize_name(anonymized.last_name)
        
        # Anonymize address
        if self.anonymize_addresses and anonymized.address:
            anonymized.address = self._anonymize_address(anonymized.address)
        
        # Anonymize phone
        if self.anonymize_phones and anonymized.phone:
            anonymized.phone = self._anonymize_phone(anonymized.phone)
        
        # Anonymize patient ID if it contains PHI
        if anonymized.patient_id:
            anonymized.patient_id = self._anonymize_id(anonymized.patient_id)
        
        # Anonymize dates (shift by random amount)
        if self.anonymize_dates:
            if anonymized.date_of_birth:
                anonymized.date_of_birth = self._shift_date(anonymized.date_of_birth)
            
            if anonymized.service_info and anonymized.service_info.date_of_service:
                anonymized.service_info.date_of_service = self._shift_date(
                    anonymized.service_info.date_of_service
                )
        
        # Mark as anonymized
        anonymized.is_anonymized = True
        
        return anonymized
    
    def _anonymize_name(self, name: str) -> str:
        """Anonymize a name."""
        return f"Patient_{hash(name) % 10000:04d}"
    
    def _anonymize_address(self, address: str) -> str:
        """Anonymize an address."""
        # Keep general geographic area but remove specific details
        words = address.split()
        anonymized_words = []
        
        for word in words:
            if re.match(r'\d+', word):  # Street numbers
                anonymized_words.append("XXX")
            elif len(word) > 3:  # Street names
                anonymized_words.append(word[:3] + "***")
            else:
                anonymized_words.append(word)
        
        return " ".join(anonymized_words)
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize a phone number."""
        # Keep area code, anonymize rest
        clean_phone = re.sub(r'[^\d]', '', phone)
        if len(clean_phone) >= 10:
            return f"({clean_phone[:3]}) XXX-XXXX"
        return "XXX-XXXX"
    
    def _anonymize_id(self, patient_id: str) -> str:
        """Anonymize a patient ID."""
        return f"ID_{hash(patient_id) % 100000:05d}"
    
    def _shift_date(self, original_date: date) -> date:
        """Shift date by random amount while preserving relative relationships."""
        import random
        
        # Shift by +/- 30 days randomly
        shift_days = random.randint(-30, 30)
        
        try:
            from datetime import timedelta
            return original_date + timedelta(days=shift_days)
        except:
            return original_date  # Return original if shift fails


class DataValidator:
    """Main data validator coordinating all validation components."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize data validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize validators
        self.cpt_validator = CPTCodeValidator(config)
        self.icd10_validator = ICD10CodeValidator(config)
        self.date_validator = DateValidator(config)
        self.phi_anonymizer = PHIAnonymizer(config)
    
    def validate_patient_data(self, patient_data: PatientData) -> Dict[str, Any]:
        """
        Perform comprehensive validation of patient data.
        
        Args:
            patient_data: Patient data to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'patient_index': patient_data.patient_index,
            'overall_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'cpt_validation': [],
            'icd10_validation': [],
            'date_validation': {},
            'phi_detected': patient_data.phi_detected
        }
        
        # Validate CPT codes
        if patient_data.cpt_codes:
            results['cpt_validation'] = self.cpt_validator.validate_cpt_codes(patient_data.cpt_codes)
            
            # Check for invalid CPT codes
            invalid_cpts = [r for r in results['cpt_validation'] if not r['is_valid']]
            if invalid_cpts:
                results['overall_valid'] = False
                results['validation_errors'].extend([
                    f"Invalid CPT code: {r['code']} - {r['message']}" 
                    for r in invalid_cpts
                ])
        
        # Validate ICD-10 codes
        if patient_data.icd10_codes:
            results['icd10_validation'] = self.icd10_validator.validate_icd10_codes(patient_data.icd10_codes)
            
            # Check for invalid ICD-10 codes
            invalid_icds = [r for r in results['icd10_validation'] if not r['is_valid']]
            if invalid_icds:
                results['overall_valid'] = False
                results['validation_errors'].extend([
                    f"Invalid ICD-10 code: {r['code']} - {r['message']}"
                    for r in invalid_icds
                ])
        
        # Validate dates
        if patient_data.date_of_birth:
            is_valid, message = self.date_validator.validate_date_of_birth(patient_data.date_of_birth)
            results['date_validation']['date_of_birth'] = {
                'is_valid': is_valid,
                'message': message
            }
            
            if not is_valid:
                results['overall_valid'] = False
                results['validation_errors'].append(f"Date of birth validation failed: {message}")
        
        # Validate service date
        if patient_data.service_info and patient_data.service_info.date_of_service:
            is_valid, message = self.date_validator.validate_service_date(
                patient_data.service_info.date_of_service,
                patient_data.date_of_birth
            )
            results['date_validation']['service_date'] = {
                'is_valid': is_valid,
                'message': message
            }
            
            if not is_valid:
                results['overall_valid'] = False
                results['validation_errors'].append(f"Service date validation failed: {message}")
        
        # Check for missing required data
        if not patient_data.cpt_codes and not patient_data.icd10_codes:
            results['validation_warnings'].append("No medical codes (CPT or ICD-10) found")
        
        if not patient_data.first_name and not patient_data.last_name:
            results['validation_warnings'].append("No patient name found")
        
        return results
    
    def validate_multiple_patients(self, patients: List[PatientData]) -> List[Dict[str, Any]]:
        """
        Validate multiple patient records.
        
        Args:
            patients: List of patient data to validate
            
        Returns:
            List of validation results
        """
        return [self.validate_patient_data(patient) for patient in patients]
    
    def anonymize_patient_data(self, patient_data: PatientData) -> PatientData:
        """
        Anonymize patient data.
        
        Args:
            patient_data: Patient data to anonymize
            
        Returns:
            Anonymized patient data
        """
        return self.phi_anonymizer.anonymize_patient_data(patient_data)
