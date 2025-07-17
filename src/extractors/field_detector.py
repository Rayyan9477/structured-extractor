"""
Field Detection System for Medical Superbill Extraction

Implements regular expressions and pattern matching for CPT codes, ICD-10 codes,
dates, PHI identification, and other medical billing fields.
"""

import re
import dateparser
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import (
    CPTCode, ICD10Code, PHIItem, PHIType, 
    FieldConfidence, Address, ContactInfo
)


class FieldType(Enum):
    """Types of fields to detect."""
    CPT_CODE = "cpt_code"
    ICD10_CODE = "icd10_code"
    DATE = "date"
    PHONE = "phone"
    EMAIL = "email"
    SSN = "ssn"
    NAME = "name"
    ADDRESS = "address"
    MONEY = "money"
    NPI = "npi"


@dataclass
class DetectionResult:
    """Result of field detection."""
    field_type: FieldType
    value: str
    confidence: float
    position: Optional[Tuple[int, int]] = None  # Start, end positions in text
    context: Optional[str] = None  # Surrounding text for validation
    metadata: Optional[Dict[str, Any]] = None


class PatternDetector:
    """Base class for pattern-based field detection."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize pattern detector.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load medical codes configuration
        self.medical_codes_config = config.get_medical_codes_config()
        
        # Load security configuration for PHI patterns
        self.security_config = config.get_security_config()
        self.phi_patterns = self.security_config.get("phi_patterns", {})
    
    def detect_fields(self, text: str) -> List[DetectionResult]:
        """
        Detect all fields in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detection results
        """
        results = []
        
        # Detect different field types
        results.extend(self.detect_cpt_codes(text))
        results.extend(self.detect_icd10_codes(text))
        results.extend(self.detect_dates(text))
        results.extend(self.detect_phi(text))
        results.extend(self.detect_financial_amounts(text))
        results.extend(self.detect_npi_numbers(text))
        
        return results
    
    def detect_cpt_codes(self, text: str) -> List[DetectionResult]:
        """Detect CPT codes in text."""
        results = []
        
        # Get CPT code pattern from config
        cpt_config = self.medical_codes_config.get("cpt_codes", {})
        pattern = cpt_config.get("pattern", r"\b\d{5}\b")
        
        # Find all matches
        for match in re.finditer(pattern, text):
            code = match.group()
            
            # Validate CPT code
            if self._is_valid_cpt_code(code):
                confidence = self._calculate_cpt_confidence(code, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.CPT_CODE,
                    value=code,
                    confidence=confidence,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return results
    
    def detect_icd10_codes(self, text: str) -> List[DetectionResult]:
        """Detect ICD-10 diagnosis codes in text."""
        results = []
        
        # Get ICD-10 pattern from config
        icd10_config = self.medical_codes_config.get("icd10_codes", {})
        pattern = icd10_config.get("pattern", r"\b[A-Z]\d{2}(\.[\dA-Z]{1,3})?\b")
        
        # Find all matches
        for match in re.finditer(pattern, text):
            code = match.group()
            
            # Validate ICD-10 code
            if self._is_valid_icd10_code(code):
                confidence = self._calculate_icd10_confidence(code, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.ICD10_CODE,
                    value=code,
                    confidence=confidence,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return results
    
    def detect_dates(self, text: str) -> List[DetectionResult]:
        """Detect dates in various formats."""
        results = []
        
        # Common date patterns
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY, MM-DD-YYYY
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2}\b",    # MM/DD/YY, MM-DD-YY
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",    # YYYY/MM/DD, YYYY-MM-DD
            r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b",  # Month DD, YYYY
            r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",    # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date_str = match.group()
                
                # Try to parse the date
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    confidence = self._calculate_date_confidence(date_str, text, match.start())
                    
                    results.append(DetectionResult(
                        field_type=FieldType.DATE,
                        value=parsed_date.isoformat(),
                        confidence=confidence,
                        position=(match.start(), match.end()),
                        context=self._get_context(text, match.start(), match.end()),
                        metadata={"original_format": date_str}
                    ))
        
        return results
    
    def detect_phi(self, text: str) -> List[DetectionResult]:
        """Detect Protected Health Information."""
        results = []
        
        # SSN detection
        ssn_pattern = self.phi_patterns.get("ssn", r"\b\d{3}-?\d{2}-?\d{4}\b")
        for match in re.finditer(ssn_pattern, text):
            results.append(DetectionResult(
                field_type=FieldType.SSN,
                value=match.group(),
                confidence=0.9,
                position=(match.start(), match.end()),
                context=self._get_context(text, match.start(), match.end())
            ))
        
        # Phone number detection
        phone_pattern = self.phi_patterns.get("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
        for match in re.finditer(phone_pattern, text):
            phone = match.group()
            if self._is_valid_phone(phone):
                results.append(DetectionResult(
                    field_type=FieldType.PHONE,
                    value=phone,
                    confidence=0.8,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        # Email detection
        email_pattern = self.phi_patterns.get("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        for match in re.finditer(email_pattern, text):
            results.append(DetectionResult(
                field_type=FieldType.EMAIL,
                value=match.group(),
                confidence=0.85,
                position=(match.start(), match.end()),
                context=self._get_context(text, match.start(), match.end())
            ))
        
        return results
    
    def detect_financial_amounts(self, text: str) -> List[DetectionResult]:
        """Detect monetary amounts."""
        results = []
        
        # Money patterns
        money_patterns = [
            r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # $1,234.56
            r"\d{1,3}(?:,\d{3})*\.\d{2}",        # 1,234.56
            r"\$\d+\.\d{2}",                      # $123.45
        ]
        
        for pattern in money_patterns:
            for match in re.finditer(pattern, text):
                amount = match.group()
                confidence = self._calculate_money_confidence(amount, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.MONEY,
                    value=amount,
                    confidence=confidence,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return results
    
    def detect_npi_numbers(self, text: str) -> List[DetectionResult]:
        """Detect NPI (National Provider Identifier) numbers."""
        results = []
        
        # NPI is exactly 10 digits
        npi_pattern = r"\b\d{10}\b"
        
        for match in re.finditer(npi_pattern, text):
            npi = match.group()
            
            # Additional validation for NPI
            if self._is_valid_npi(npi):
                confidence = self._calculate_npi_confidence(npi, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.NPI,
                    value=npi,
                    confidence=confidence,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return results
    
    def _is_valid_cpt_code(self, code: str) -> bool:
        """Validate CPT code format and range."""
        if not re.match(r"^\d{5}$", code):
            return False
        
        # CPT codes range from 00100 to 99999
        code_num = int(code)
        return 100 <= code_num <= 99999
    
    def _is_valid_icd10_code(self, code: str) -> bool:
        """Validate ICD-10 code format."""
        # Basic format validation
        if not re.match(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,3})?$", code):
            return False
        
        # Check if first character is valid ICD-10 chapter
        first_char = code[0]
        valid_chapters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return first_char in valid_chapters
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number."""
        # Remove formatting
        digits = re.sub(r"[^\d]", "", phone)
        return len(digits) == 10
    
    def _is_valid_npi(self, npi: str) -> bool:
        """Validate NPI using Luhn algorithm."""
        if len(npi) != 10:
            return False
        
        # NPI Luhn check
        # Add prefix "80840" to the 10-digit NPI
        full_npi = "80840" + npi
        
        # Apply Luhn algorithm
        def luhn_check(num_str):
            digits = [int(d) for d in num_str]
            checksum = 0
            
            # Process from right to left
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:  # Every second digit from right
                    digit *= 2
                    if digit > 9:
                        digit = digit // 10 + digit % 10
                checksum += digit
            
            return checksum % 10 == 0
        
        return luhn_check(full_npi)
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string using dateparser."""
        try:
            parsed = dateparser.parse(
                date_str,
                settings={
                    'STRICT_PARSING': False,
                    'DATE_ORDER': 'MDY',  # US date format preference
                    'RETURN_AS_TIMEZONE_AWARE': False
                }
            )
            return parsed.date() if parsed else None
        except Exception:
            return None
    
    def _calculate_cpt_confidence(self, code: str, text: str, position: int) -> float:
        """Calculate confidence score for CPT code detection."""
        base_confidence = 0.8
        
        # Check context for medical terms
        context = self._get_context(text, position, position + len(code), window=50)
        medical_keywords = [
            "procedure", "treatment", "exam", "visit", "consultation",
            "surgery", "therapy", "injection", "test", "screening"
        ]
        
        if any(keyword in context.lower() for keyword in medical_keywords):
            base_confidence += 0.1
        
        # Check if it's a known valid CPT code range
        code_num = int(code)
        if 99200 <= code_num <= 99499:  # E&M codes
            base_confidence += 0.1
        elif 10000 <= code_num <= 69999:  # Surgery codes
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_icd10_confidence(self, code: str, text: str, position: int) -> float:
        """Calculate confidence score for ICD-10 code detection."""
        base_confidence = 0.8
        
        # Check context for diagnosis terms
        context = self._get_context(text, position, position + len(code), window=50)
        diagnosis_keywords = [
            "diagnosis", "condition", "disease", "disorder", "syndrome",
            "injury", "pain", "infection", "primary", "secondary"
        ]
        
        if any(keyword in context.lower() for keyword in diagnosis_keywords):
            base_confidence += 0.1
        
        # Check if it has decimal (more specific codes have higher confidence)
        if "." in code:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_date_confidence(self, date_str: str, text: str, position: int) -> float:
        """Calculate confidence score for date detection."""
        base_confidence = 0.7
        
        # Check context for date-related terms
        context = self._get_context(text, position, position + len(date_str), window=30)
        date_keywords = [
            "date", "service", "visit", "admission", "discharge",
            "birth", "dob", "seen", "treated", "exam"
        ]
        
        if any(keyword in context.lower() for keyword in date_keywords):
            base_confidence += 0.2
        
        # Higher confidence for standard formats
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _calculate_money_confidence(self, amount: str, text: str, position: int) -> float:
        """Calculate confidence score for monetary amount detection."""
        base_confidence = 0.7
        
        # Check context for financial terms
        context = self._get_context(text, position, position + len(amount), window=30)
        financial_keywords = [
            "charge", "fee", "cost", "amount", "payment", "copay",
            "deductible", "balance", "total", "bill", "invoice"
        ]
        
        if any(keyword in context.lower() for keyword in financial_keywords):
            base_confidence += 0.2
        
        # Higher confidence for properly formatted amounts
        if "$" in amount and "." in amount:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _calculate_npi_confidence(self, npi: str, text: str, position: int) -> float:
        """Calculate confidence score for NPI detection."""
        base_confidence = 0.6  # Lower base confidence as 10 digits can be many things
        
        # Check context for provider terms
        context = self._get_context(text, position, position + len(npi), window=50)
        provider_keywords = [
            "npi", "provider", "physician", "doctor", "dr", "md",
            "nurse", "therapist", "practitioner", "clinic", "practice"
        ]
        
        if any(keyword in context.lower() for keyword in provider_keywords):
            base_confidence += 0.3
        
        # Higher confidence if NPI validation passes
        if self._is_valid_npi(npi):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for a detected field."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class NameDetector:
    """Specialized detector for patient and provider names."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize name detector.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Common name patterns and prefixes
        self.name_prefixes = [
            "mr", "mrs", "ms", "dr", "doctor", "miss", "patient",
            "pt", "provider", "physician", "nurse"
        ]
        
        self.name_suffixes = [
            "jr", "sr", "ii", "iii", "iv", "md", "do", "rn", "np", "pa"
        ]
    
    def detect_names(self, text: str) -> List[DetectionResult]:
        """Detect potential patient and provider names."""
        results = []
        
        # Pattern for name detection
        # Look for capitalized words that could be names
        name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        
        for match in re.finditer(name_pattern, text):
            name = match.group()
            
            # Validate if this looks like a real name
            if self._is_likely_name(name, text, match.start()):
                confidence = self._calculate_name_confidence(name, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.NAME,
                    value=name,
                    confidence=confidence,
                    position=(match.start(), match.end()),
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return results
    
    def _is_likely_name(self, candidate: str, text: str, position: int) -> bool:
        """Check if a candidate string is likely a name."""
        # Skip single words unless they have name context
        words = candidate.split()
        if len(words) < 2:
            context = self._get_context(text, position, position + len(candidate), 20)
            return any(prefix in context.lower() for prefix in self.name_prefixes)
        
        # Skip if it contains common non-name words
        non_name_words = [
            "and", "or", "the", "of", "in", "at", "on", "for", "with",
            "by", "from", "to", "date", "code", "number", "amount"
        ]
        
        if any(word.lower() in non_name_words for word in words):
            return False
        
        # Must be reasonable length
        if len(candidate) > 50:
            return False
        
        return True
    
    def _calculate_name_confidence(self, name: str, text: str, position: int) -> float:
        """Calculate confidence score for name detection."""
        base_confidence = 0.5
        
        # Check context for name indicators
        context = self._get_context(text, position, position + len(name), 30)
        
        # Look for name prefixes/suffixes
        if any(prefix in context.lower() for prefix in self.name_prefixes):
            base_confidence += 0.3
        
        if any(suffix in name.lower() for suffix in self.name_suffixes):
            base_confidence += 0.2
        
        # Multiple words increase confidence
        word_count = len(name.split())
        if word_count == 2:
            base_confidence += 0.1
        elif word_count >= 3:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Get surrounding context for a detected field."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class AddressDetector:
    """Specialized detector for addresses."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize address detector.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # US state abbreviations
        self.states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
        ]
        
        # Common street suffixes
        self.street_suffixes = [
            "st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road",
            "dr", "drive", "ln", "lane", "ct", "court", "pl", "place",
            "way", "circle", "cir", "pkwy", "parkway"
        ]
    
    def detect_addresses(self, text: str) -> List[DetectionResult]:
        """Detect addresses in text."""
        results = []
        
        # Pattern for street addresses
        street_pattern = r"\d+\s+[A-Za-z\s]+(?:" + "|".join(self.street_suffixes) + r")\b"
        
        for match in re.finditer(street_pattern, text, re.IGNORECASE):
            address_part = match.group()
            
            # Look for complete address (with city, state, zip)
            extended_match = self._find_complete_address(text, match.start())
            
            if extended_match:
                confidence = self._calculate_address_confidence(extended_match, text, match.start())
                
                results.append(DetectionResult(
                    field_type=FieldType.ADDRESS,
                    value=extended_match,
                    confidence=confidence,
                    position=(match.start(), match.start() + len(extended_match)),
                    context=self._get_context(text, match.start(), match.start() + len(extended_match))
                ))
        
        return results
    
    def _find_complete_address(self, text: str, start_pos: int) -> Optional[str]:
        """Try to find a complete address starting from a street address."""
        # Look ahead for city, state, zip pattern
        remaining_text = text[start_pos:start_pos + 200]  # Look ahead 200 chars
        
        # Pattern for city, state zip
        city_state_zip_pattern = r"([A-Za-z\s]+),?\s+(" + "|".join(self.states) + r")\s+(\d{5}(?:-\d{4})?)"
        
        match = re.search(city_state_zip_pattern, remaining_text)
        if match:
            return remaining_text[:match.end()]
        
        return None
    
    def _calculate_address_confidence(self, address: str, text: str, position: int) -> float:
        """Calculate confidence score for address detection."""
        base_confidence = 0.6
        
        # Check for state abbreviation
        if any(state in address.upper() for state in self.states):
            base_confidence += 0.2
        
        # Check for ZIP code
        if re.search(r"\d{5}(?:-\d{4})?", address):
            base_confidence += 0.1
        
        # Check for street suffix
        if any(suffix in address.lower() for suffix in self.street_suffixes):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Get surrounding context for a detected field."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]


class FieldDetectionEngine:
    """Main field detection engine that coordinates all detectors."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize field detection engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize detectors
        self.pattern_detector = PatternDetector(config)
        self.name_detector = NameDetector(config)
        self.address_detector = AddressDetector(config)
    
    def detect_all_fields(self, text: str) -> Dict[FieldType, List[DetectionResult]]:
        """
        Detect all fields in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of field types to detection results
        """
        self.logger.debug("Starting field detection")
        
        all_results = []
        
        # Run all detectors
        all_results.extend(self.pattern_detector.detect_fields(text))
        all_results.extend(self.name_detector.detect_names(text))
        all_results.extend(self.address_detector.detect_addresses(text))
        
        # Group results by field type
        grouped_results = {}
        for result in all_results:
            if result.field_type not in grouped_results:
                grouped_results[result.field_type] = []
            grouped_results[result.field_type].append(result)
        
        # Sort results by confidence within each group
        for field_type in grouped_results:
            grouped_results[field_type].sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.debug(f"Field detection completed: {len(all_results)} fields found")
        return grouped_results
    
    def get_best_matches(
        self,
        detection_results: Dict[FieldType, List[DetectionResult]],
        max_per_type: int = 5
    ) -> Dict[FieldType, List[DetectionResult]]:
        """
        Get the best matches for each field type.
        
        Args:
            detection_results: Results from detect_all_fields
            max_per_type: Maximum results per field type
            
        Returns:
            Filtered results with best matches
        """
        filtered_results = {}
        
        for field_type, results in detection_results.items():
            # Take top N results based on confidence
            filtered_results[field_type] = results[:max_per_type]
        
        return filtered_results
