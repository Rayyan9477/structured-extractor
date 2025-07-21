"""
Mixed Patient Data Validator for Medical Superbill Extraction

Provides advanced validation logic to detect signs of mixed patient data 
or when multiple patients' information has been incorrectly combined.
"""

from typing import List
from src.core.data_schema import PatientData
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class MixedPatientValidator:
    """Validates patient records for signs of mixed patient data."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize mixed patient validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate_for_mixed_patient_data(self, patient_data: PatientData) -> List[str]:
        """
        Validate a patient record to detect signs of mixed patient data.
        
        Args:
            patient_data: Patient data to validate
            
        Returns:
            List of potential issues indicating mixed patient data
        """
        issues = []
        
        # Check for unusually high number of diagnostic codes
        if hasattr(patient_data, 'icd10_codes') and patient_data.icd10_codes:
            if len(patient_data.icd10_codes) > 12:  # Arbitrary threshold, adjust based on domain knowledge
                issues.append(f"Unusually high number of ICD-10 codes: {len(patient_data.icd10_codes)}")
        
        if hasattr(patient_data, 'cpt_codes') and patient_data.cpt_codes:
            if len(patient_data.cpt_codes) > 15:  # Arbitrary threshold
                issues.append(f"Unusually high number of CPT codes: {len(patient_data.cpt_codes)}")
        
        # Check for inconsistent demographics
        if hasattr(patient_data, 'gender') and patient_data.gender:
            gender = patient_data.gender.lower()
            
            # Look for gender-specific procedures that conflict with the patient's gender
            if hasattr(patient_data, 'cpt_codes') and patient_data.cpt_codes:
                # Examples of gender-specific CPT codes (simplified)
                male_specific_prefixes = ['5438', '5439', '5463']  # Example male-specific procedure code prefixes
                female_specific_prefixes = ['5841', '5842', '5921']  # Example female-specific procedure code prefixes
                
                if gender == 'male':
                    for cpt in patient_data.cpt_codes:
                        if any(cpt.code.startswith(prefix) for prefix in female_specific_prefixes):
                            issues.append(f"Found female-specific procedure code {cpt.code} for male patient")
                
                if gender == 'female':
                    for cpt in patient_data.cpt_codes:
                        if any(cpt.code.startswith(prefix) for prefix in male_specific_prefixes):
                            issues.append(f"Found male-specific procedure code {cpt.code} for female patient")
        
        # Check for inconsistent age-related data
        age_related_issues = self._check_age_consistency(patient_data)
        issues.extend(age_related_issues)
        
        # Check for multiple name patterns in the same record
        name_issues = self._check_name_consistency(patient_data)
        issues.extend(name_issues)
        
        return issues
    
    def _check_age_consistency(self, patient_data: PatientData) -> List[str]:
        """Check for age-related inconsistencies."""
        issues = []
        
        # Check for age-inappropriate diagnoses or procedures
        if hasattr(patient_data, 'date_of_birth') and patient_data.date_of_birth:
            try:
                from datetime import datetime
                dob = datetime.strptime(patient_data.date_of_birth, "%Y-%m-%d")
                current_date = datetime.now()
                age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
                
                # Check for pediatric codes in elderly patients
                if age > 65:
                    if hasattr(patient_data, 'cpt_codes') and patient_data.cpt_codes:
                        pediatric_prefixes = ['99381', '99382', '99383', '99384']  # Example pediatric CPT codes
                        for cpt in patient_data.cpt_codes:
                            if any(cpt.code.startswith(prefix) for prefix in pediatric_prefixes):
                                issues.append(f"Found pediatric procedure code {cpt.code} for elderly patient (age {age})")
                
                # Check for geriatric codes in young patients
                if age < 18:
                    if hasattr(patient_data, 'cpt_codes') and patient_data.cpt_codes:
                        geriatric_prefixes = ['99366', '99368']  # Example geriatric CPT codes
                        for cpt in patient_data.cpt_codes:
                            if any(cpt.code.startswith(prefix) for prefix in geriatric_prefixes):
                                issues.append(f"Found geriatric procedure code {cpt.code} for young patient (age {age})")
            except:
                pass  # Skip if date parsing fails
        
        return issues
    
    def _check_name_consistency(self, patient_data: PatientData) -> List[str]:
        """Check for multiple name patterns that might indicate mixed records."""
        issues = []
        
        # Look for inconsistent name patterns
        if (hasattr(patient_data, 'first_name') and patient_data.first_name and 
            hasattr(patient_data, 'last_name') and patient_data.last_name):
            
            # Check for potential multiple names in a single field
            if ";" in patient_data.first_name or ";" in patient_data.last_name:
                issues.append("Multiple names detected in name fields")
            
            # Check for unusual patterns like multiple spaces or conjunctions
            if " and " in patient_data.first_name or " & " in patient_data.first_name:
                issues.append("Possible multiple patients in first name field")
                
            if " and " in patient_data.last_name or " & " in patient_data.last_name:
                issues.append("Possible multiple patients in last name field")
        
        return issues
