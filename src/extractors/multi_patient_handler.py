"""
Multi-Patient Handling System for Medical Superbill Extraction

Implements patient boundary detection algorithms, data segregation by patient,
cross-validation of patient-specific data, and duplicate detection.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import difflib

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import PatientData, SuperbillDocument
from src.extractors.field_detector import FieldType, DetectionResult


class BoundaryType(Enum):
    """Types of patient boundaries."""
    KEYWORD_SEPARATOR = "keyword_separator"
    FORM_STRUCTURE = "form_structure"
    TEXT_BLOCK = "text_block"
    PATTERN_REPEAT = "pattern_repeat"
    PAGE_BREAK = "page_break"


@dataclass
class PatientBoundary:
    """Represents a boundary between patients."""
    start_position: int
    end_position: Optional[int]
    boundary_type: BoundaryType
    confidence: float
    keywords_found: List[str]
    patient_index: int


@dataclass
class PatientSegment:
    """Represents a text segment belonging to a specific patient."""
    patient_index: int
    text: str
    start_position: int
    end_position: int
    confidence: float
    boundary_info: PatientBoundary
    extracted_data: Optional[PatientData] = None


class PatientBoundaryDetector:
    """Detects boundaries between different patients in a document."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize patient boundary detector.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get patient detection configuration
        self.patient_config = config.get("patient_detection", {})
        self.max_patients = self.patient_config.get("max_patients_per_document", 10)
        self.confidence_threshold = self.patient_config.get("confidence_threshold", 0.8)
        self.separation_keywords = self.patient_config.get("separation_keywords", [
            "PATIENT", "NAME", "DOB", "ACCOUNT", "CLAIM", "PT", "PATIENT NAME"
        ])
        
        # Compile keyword patterns
        self.keyword_patterns = [
            re.compile(rf"\b{keyword}\b", re.IGNORECASE) 
            for keyword in self.separation_keywords
        ]
    
    def detect_patient_boundaries(self, text: str) -> List[PatientBoundary]:
        """
        Detect patient boundaries in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected patient boundaries
        """
        self.logger.debug("Detecting patient boundaries")
        
        boundaries = []
        
        # Method 1: Keyword-based detection
        boundaries.extend(self._detect_keyword_boundaries(text))
        
        # Method 2: Pattern repetition detection
        boundaries.extend(self._detect_pattern_boundaries(text))
        
        # Method 3: Form structure detection
        boundaries.extend(self._detect_form_boundaries(text))
        
        # Method 4: Line-based detection
        boundaries.extend(self._detect_line_boundaries(text))
        
        # Filter and validate boundaries
        validated_boundaries = self._validate_boundaries(boundaries, text)
        
        # Sort by position
        validated_boundaries.sort(key=lambda b: b.start_position)
        
        # Assign patient indices
        for i, boundary in enumerate(validated_boundaries):
            boundary.patient_index = i
        
        self.logger.debug(f"Detected {len(validated_boundaries)} patient boundaries")
        return validated_boundaries
    
    def _detect_keyword_boundaries(self, text: str) -> List[PatientBoundary]:
        """Detect boundaries based on separation keywords."""
        boundaries = []
        
        for pattern in self.keyword_patterns:
            for match in pattern.finditer(text):
                # Look for additional context around the keyword
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                # Calculate confidence based on context
                confidence = self._calculate_keyword_confidence(match.group(), context)
                
                if confidence >= self.confidence_threshold:
                    boundaries.append(PatientBoundary(
                        start_position=match.start(),
                        end_position=None,
                        boundary_type=BoundaryType.KEYWORD_SEPARATOR,
                        confidence=confidence,
                        keywords_found=[match.group()],
                        patient_index=0  # Will be set later
                    ))
        
        return boundaries
    
    def _detect_pattern_boundaries(self, text: str) -> List[PatientBoundary]:
        """Detect boundaries based on repeating patterns."""
        boundaries = []
        
        # Common patterns that repeat for each patient
        patterns = [
            r"Name:\s*[A-Za-z\s]+",
            r"DOB:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"Patient\s+ID:\s*\w+",
            r"Account\s*#:\s*\w+",
            r"Service\s+Date:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if len(matches) > 1:  # Multiple occurrences suggest multiple patients
                for match in matches:
                    confidence = 0.7  # Base confidence for pattern matches
                    
                    boundaries.append(PatientBoundary(
                        start_position=match.start(),
                        end_position=None,
                        boundary_type=BoundaryType.PATTERN_REPEAT,
                        confidence=confidence,
                        keywords_found=[match.group()],
                        patient_index=0
                    ))
        
        return boundaries
    
    def _detect_form_boundaries(self, text: str) -> List[PatientBoundary]:
        """Detect boundaries based on form-like structures."""
        boundaries = []
        
        # Look for horizontal lines or separators
        line_patterns = [
            r"-{10,}",  # Horizontal dashes
            r"={10,}",  # Horizontal equals
            r"_{10,}",  # Horizontal underscores
            r"\*{10,}", # Horizontal asterisks
        ]
        
        for pattern in line_patterns:
            for match in re.finditer(pattern, text):
                # Check if this line might separate patient sections
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                
                line_content = text[line_start:line_end]
                
                # Only consider lines that are mostly separators
                if len(match.group()) / len(line_content.strip()) > 0.7:
                    boundaries.append(PatientBoundary(
                        start_position=match.start(),
                        end_position=line_end,
                        boundary_type=BoundaryType.FORM_STRUCTURE,
                        confidence=0.6,
                        keywords_found=[],
                        patient_index=0
                    ))
        
        return boundaries
    
    def _detect_line_boundaries(self, text: str) -> List[PatientBoundary]:
        """Detect boundaries based on line patterns and spacing."""
        boundaries = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for lines that might indicate new patient
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check if line contains patient indicators
            patient_indicators = ['patient', 'pt', 'name', 'dob', 'account']
            
            if any(indicator in line_stripped.lower() for indicator in patient_indicators):
                # Calculate position in original text
                position = sum(len(lines[j]) + 1 for j in range(i))  # +1 for newline
                
                confidence = self._calculate_line_confidence(line_stripped)
                
                if confidence >= 0.5:
                    boundaries.append(PatientBoundary(
                        start_position=position,
                        end_position=None,
                        boundary_type=BoundaryType.TEXT_BLOCK,
                        confidence=confidence,
                        keywords_found=[line_stripped],
                        patient_index=0
                    ))
        
        return boundaries
    
    def _calculate_keyword_confidence(self, keyword: str, context: str) -> float:
        """Calculate confidence for keyword-based boundary detection."""
        base_confidence = 0.7
        
        # Higher confidence for exact keyword matches
        if keyword.upper() in ["PATIENT", "NAME", "DOB"]:
            base_confidence += 0.2
        
        # Check for additional patient-related terms in context
        patient_terms = ['name', 'dob', 'birth', 'address', 'phone', 'id', 'account']
        term_count = sum(1 for term in patient_terms if term in context.lower())
        
        base_confidence += min(term_count * 0.05, 0.2)
        
        return min(base_confidence, 1.0)
    
    def _calculate_line_confidence(self, line: str) -> float:
        """Calculate confidence for line-based boundary detection."""
        base_confidence = 0.5
        
        # Check for colon patterns (field: value)
        if ':' in line:
            base_confidence += 0.2
        
        # Check for capitalization patterns
        if line.isupper():
            base_confidence += 0.1
        
        # Check for specific field patterns
        field_patterns = [
            r'name.*:',
            r'dob.*:',
            r'patient.*:',
            r'account.*:',
            r'id.*:'
        ]
        
        for pattern in field_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                base_confidence += 0.15
                break
        
        return min(base_confidence, 1.0)
    
    def _validate_boundaries(self, boundaries: List[PatientBoundary], text: str) -> List[PatientBoundary]:
        """Validate and filter boundaries."""
        if not boundaries:
            return []
        
        # Remove duplicates (boundaries too close to each other)
        filtered_boundaries = []
        
        # Sort by position
        boundaries.sort(key=lambda b: b.start_position)
        
        for boundary in boundaries:
            # Check if this boundary is too close to existing ones
            too_close = False
            for existing in filtered_boundaries:
                if abs(boundary.start_position - existing.start_position) < 100:
                    # Keep the one with higher confidence
                    if boundary.confidence > existing.confidence:
                        filtered_boundaries.remove(existing)
                    else:
                        too_close = True
                    break
            
            if not too_close:
                filtered_boundaries.append(boundary)
        
        # Limit to maximum number of patients
        filtered_boundaries = filtered_boundaries[:self.max_patients - 1]  # -1 because first patient doesn't need boundary
        
        return filtered_boundaries


class PatientSegmenter:
    """Segments text into patient-specific sections."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize patient segmenter.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.boundary_detector = PatientBoundaryDetector(config)
    
    def segment_text(self, text: str) -> List[PatientSegment]:
        """
        Segment text into patient-specific sections.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of patient segments
        """
        self.logger.debug("Segmenting text by patient")
        
        # Detect boundaries
        boundaries = self.boundary_detector.detect_patient_boundaries(text)
        
        # Create segments
        segments = []
        
        if not boundaries:
            # Single patient document
            segments.append(PatientSegment(
                patient_index=0,
                text=text,
                start_position=0,
                end_position=len(text),
                confidence=1.0,
                boundary_info=PatientBoundary(
                    start_position=0,
                    end_position=len(text),
                    boundary_type=BoundaryType.TEXT_BLOCK,
                    confidence=1.0,
                    keywords_found=[],
                    patient_index=0
                )
            ))
        else:
            # Multi-patient document
            
            # First segment (before first boundary)
            if boundaries[0].start_position > 0:
                segments.append(PatientSegment(
                    patient_index=0,
                    text=text[:boundaries[0].start_position],
                    start_position=0,
                    end_position=boundaries[0].start_position,
                    confidence=0.8,
                    boundary_info=boundaries[0]
                ))
            
            # Segments between boundaries
            for i, boundary in enumerate(boundaries):
                start_pos = boundary.start_position
                end_pos = boundaries[i + 1].start_position if i + 1 < len(boundaries) else len(text)
                
                segments.append(PatientSegment(
                    patient_index=i + 1,
                    text=text[start_pos:end_pos],
                    start_position=start_pos,
                    end_position=end_pos,
                    confidence=boundary.confidence,
                    boundary_info=boundary
                ))
        
        # Filter out segments that are too short
        segments = [s for s in segments if len(s.text.strip()) > 50]
        
        # Re-index patients
        for i, segment in enumerate(segments):
            segment.patient_index = i
            segment.boundary_info.patient_index = i
        
        self.logger.debug(f"Created {len(segments)} patient segments")
        return segments


class PatientDataValidator:
    """Validates and cross-references patient data."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize patient data validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def validate_patient_data(self, patient_data: PatientData) -> Tuple[bool, List[str]]:
        """
        Validate patient data for completeness and consistency.
        
        Args:
            patient_data: Patient data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not patient_data.first_name and not patient_data.last_name:
            errors.append("Patient name is missing")
        
        # Validate CPT codes
        for cpt in patient_data.cpt_codes:
            if not self._validate_cpt_code(cpt.code):
                errors.append(f"Invalid CPT code: {cpt.code}")
        
        # Validate ICD-10 codes
        for icd in patient_data.icd10_codes:
            if not self._validate_icd10_code(icd.code):
                errors.append(f"Invalid ICD-10 code: {icd.code}")
        
        # Validate dates
        if patient_data.date_of_birth:
            from datetime import date
            if patient_data.date_of_birth > date.today():
                errors.append("Date of birth is in the future")
        
        # Check data consistency
        if patient_data.service_info and patient_data.service_info.date_of_service:
            if patient_data.date_of_birth:
                # Check if service date is reasonable relative to birth date
                age_at_service = (patient_data.service_info.date_of_service - patient_data.date_of_birth).days / 365.25
                if age_at_service < 0:
                    errors.append("Service date is before birth date")
                elif age_at_service > 150:
                    errors.append("Patient age at service is unrealistic")
        
        return len(errors) == 0, errors
    
    def _validate_cpt_code(self, code: str) -> bool:
        """Validate CPT code format."""
        return bool(re.match(r'^\d{5}$', code)) and 100 <= int(code) <= 99999
    
    def _validate_icd10_code(self, code: str) -> bool:
        """Validate ICD-10 code format."""
        return bool(re.match(r'^[A-Z]\d{2}(\.[A-Z0-9]{1,3})?$', code))
    
    def detect_duplicates(self, patients: List[PatientData]) -> List[Tuple[int, int, float]]:
        """
        Detect potential duplicate patients.
        
        Args:
            patients: List of patient data
            
        Returns:
            List of tuples (patient1_index, patient2_index, similarity_score)
        """
        duplicates = []
        
        for i in range(len(patients)):
            for j in range(i + 1, len(patients)):
                similarity = self._calculate_patient_similarity(patients[i], patients[j])
                
                if similarity > 0.8:  # High similarity threshold
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def _calculate_patient_similarity(self, patient1: PatientData, patient2: PatientData) -> float:
        """Calculate similarity between two patients."""
        similarities = []
        
        # Name similarity
        name1 = f"{patient1.first_name or ''} {patient1.last_name or ''}".strip()
        name2 = f"{patient2.first_name or ''} {patient2.last_name or ''}".strip()
        
        if name1 and name2:
            name_sim = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            similarities.append(name_sim * 0.4)  # 40% weight
        
        # DOB similarity
        if patient1.date_of_birth and patient2.date_of_birth:
            dob_sim = 1.0 if patient1.date_of_birth == patient2.date_of_birth else 0.0
            similarities.append(dob_sim * 0.3)  # 30% weight
        
        # Patient ID similarity
        if patient1.patient_id and patient2.patient_id:
            id_sim = 1.0 if patient1.patient_id == patient2.patient_id else 0.0
            similarities.append(id_sim * 0.3)  # 30% weight
        
        return sum(similarities) / len(similarities) if similarities else 0.0


class MultiPatientHandler:
    """Main handler for multi-patient document processing."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize multi-patient handler.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.segmenter = PatientSegmenter(config)
        self.validator = PatientDataValidator(config)
    
    async def process_multi_patient_document(
        self, 
        text: str, 
        extractor_func
    ) -> List[PatientData]:
        """
        Process a multi-patient document.
        
        Args:
            text: Input text containing multiple patients
            extractor_func: Function to extract patient data from text segment
            
        Returns:
            List of extracted patient data
        """
        self.logger.info("Processing multi-patient document")
        
        # Segment text by patient
        segments = self.segmenter.segment_text(text)
        
        # Extract data for each patient segment
        patient_data_list = []
        
        for segment in segments:
            self.logger.debug(f"Processing patient segment {segment.patient_index}")
            
            try:
                # Extract patient data
                patient_data = await extractor_func(segment.text, segment.patient_index)
                
                if patient_data:
                    # Validate data
                    is_valid, errors = self.validator.validate_patient_data(patient_data)
                    
                    if is_valid:
                        patient_data_list.append(patient_data)
                    else:
                        self.logger.warning(f"Patient {segment.patient_index} validation failed: {errors}")
                        # Still add the data but mark validation errors
                        patient_data.validation_errors = errors
                        patient_data_list.append(patient_data)
                
            except Exception as e:
                self.logger.error(f"Failed to extract data for patient {segment.patient_index}: {e}")
        
        # Check for duplicates
        duplicates = self.validator.detect_duplicates(patient_data_list)
        
        if duplicates:
            self.logger.warning(f"Detected {len(duplicates)} potential duplicate patients")
            patient_data_list = self._merge_duplicates(patient_data_list, duplicates)
        
        self.logger.info(f"Successfully processed {len(patient_data_list)} patients")
        return patient_data_list
    
    def _merge_duplicates(
        self, 
        patients: List[PatientData], 
        duplicates: List[Tuple[int, int, float]]
    ) -> List[PatientData]:
        """Merge duplicate patient records."""
        # Simple approach: remove duplicates with lower confidence
        # In production, you might want more sophisticated merging
        
        to_remove = set()
        
        for i, j, similarity in duplicates:
            if i not in to_remove and j not in to_remove:
                # Keep the one with higher extraction confidence
                if patients[i].extraction_confidence >= patients[j].extraction_confidence:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
        
        # Remove duplicates
        merged_patients = [
            patient for i, patient in enumerate(patients) 
            if i not in to_remove
        ]
        
        return merged_patients
