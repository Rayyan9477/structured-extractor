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
        
        # Enhanced list of separation keywords for better patient identification
        self.separation_keywords = self.patient_config.get("separation_keywords", [
            "PATIENT", "NAME", "DOB", "ACCOUNT", "CLAIM", "PT", "PATIENT NAME", 
            "PATIENT ID", "MEDICAL RECORD", "MRN", "INSURANCE", "POLICY",
            "PATIENT INFORMATION", "DEMOGRAPHICS", "DATE OF BIRTH"
        ])
        
        # Additional patterns for patient identification
        self.patient_id_patterns = [
            r"(?:MRN|ID)[:.\s]*([A-Za-z0-9-]+)",
            r"(?:PATIENT|PT)[\s#]+([A-Za-z0-9-]+)",
            r"ACCOUNT[\s#:]+([A-Za-z0-9-]+)"
        ]
        
        self.name_patterns = [
            r"(?:PATIENT|PT|NAME)[:.\s]+((?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+)",
            r"(?:LAST|FIRST)[:.\s]+((?:[A-Z][a-z]+\s*)+)"
        ]
        
        # Compile keyword patterns
        self.keyword_patterns = [
            re.compile(rf"\b{keyword}\b", re.IGNORECASE) 
            for keyword in self.separation_keywords
        ]
        
        # Compile identification patterns
        self.id_patterns = [re.compile(p, re.IGNORECASE) for p in self.patient_id_patterns]
        self.patient_name_patterns = [re.compile(p, re.IGNORECASE) for p in self.name_patterns]
    
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
        
        # Method 2: Patient identifier detection (new)
        boundaries.extend(self._detect_patient_identifiers(text))
        
        # Method 3: Pattern repetition detection
        boundaries.extend(self._detect_pattern_boundaries(text))
        
        # Method 4: Form structure detection
        boundaries.extend(self._detect_form_boundaries(text))
        
        # Method 5: Line-based detection
        boundaries.extend(self._detect_line_boundaries(text))
        
        # Initial validation of boundaries
        validated_boundaries = self._validate_boundaries(boundaries, text)
        
        # If we don't have enough boundaries or they don't look reliable,
        # apply enhanced detection methods
        if len(validated_boundaries) < 1 or (
            len(validated_boundaries) == 1 and validated_boundaries[0].confidence < 0.8
        ):
            self.logger.debug("Using enhanced boundary detection methods")
            validated_boundaries = self._enhance_boundary_detection(text, validated_boundaries)
        
        # Sort by position
        validated_boundaries.sort(key=lambda b: b.start_position)
        
        # Assign patient indices
        for i, boundary in enumerate(validated_boundaries):
            boundary.patient_index = i
        
        self.logger.debug(f"Detected {len(validated_boundaries)} patient boundaries")
        return validated_boundaries
        
    def _detect_patient_identifiers(self, text: str) -> List[PatientBoundary]:
        """
        Detect patient boundaries based on patient identifiers like MRN or patient ID.
        
        Args:
            text: Input text
            
        Returns:
            List of detected boundaries
        """
        boundaries = []
        
        # Find patient IDs
        for pattern in self.id_patterns:
            for match in pattern.finditer(text):
                patient_id = match.group(1) if match.groups() else None
                if patient_id:
                    # Look for context around the ID
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    
                    # If this context looks like a new patient section, add boundary
                    if self._context_suggests_new_patient(context):
                        boundaries.append(PatientBoundary(
                            start_position=match.start(),
                            end_position=None,
                            boundary_type=BoundaryType.FORM_STRUCTURE,
                            confidence=0.85,  # High confidence for patient IDs
                            keywords_found=[match.group(0)],
                            patient_index=0  # Will be set later
                        ))
        
        # Find patient names
        for pattern in self.patient_name_patterns:
            for match in pattern.finditer(text):
                name = match.group(1) if match.groups() else None
                if name:
                    # Look for context around the name
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    
                    # If this context looks like a new patient section, add boundary
                    if self._context_suggests_new_patient(context):
                        boundaries.append(PatientBoundary(
                            start_position=match.start(),
                            end_position=None,
                            boundary_type=BoundaryType.FORM_STRUCTURE,
                            confidence=0.8,
                            keywords_found=[match.group(0)],
                            patient_index=0  # Will be set later
                        ))
        
        return boundaries
    
    def _context_suggests_new_patient(self, context: str) -> bool:
        """
        Check if the context suggests a new patient section.
        
        Args:
            context: Text context around a potential boundary
            
        Returns:
            True if context suggests a new patient section
        """
        # Check for demographic indicators
        demographic_indicators = [
            r"\bDOB\b", r"\bBIRTH\b", r"\bAGE\b", r"\bSEX\b", r"\bGENDER\b",
            r"\bADDRESS\b", r"\bPHONE\b", r"\bEMAIL\b", r"\bINSURANCE\b"
        ]
        
        indicator_count = 0
        for indicator in demographic_indicators:
            if re.search(indicator, context, re.IGNORECASE):
                indicator_count += 1
        
        # If at least 2 demographic indicators are present, likely a new patient section
        return indicator_count >= 2
    
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
    
    def _enhance_boundary_detection(self, text: str, initial_boundaries: List[PatientBoundary]) -> List[PatientBoundary]:
        """
        Enhance boundary detection with advanced techniques.
        
        Args:
            text: Input text
            initial_boundaries: Initially detected boundaries
            
        Returns:
            Enhanced list of boundaries
        """
        # If we already have reasonable boundaries, enhance them
        if initial_boundaries and len(initial_boundaries) >= 2:
            return initial_boundaries
        
        # Otherwise, try more aggressive detection methods
        enhanced_boundaries = []
        
        # Method 1: Look for page breaks
        page_breaks = re.finditer(r"---\s*PAGE\s*BREAK\s*---", text, re.IGNORECASE)
        for match in page_breaks:
            # For each page break, check if it's likely a patient boundary
            context_before = text[max(0, match.start() - 200):match.start()]
            context_after = text[match.end():min(len(text), match.end() + 200)]
            
            # Check if contexts look like different patients
            if self._contexts_suggest_different_patients(context_before, context_after):
                enhanced_boundaries.append(PatientBoundary(
                    start_position=match.start(),
                    end_position=match.end(),
                    boundary_type=BoundaryType.PAGE_BREAK,
                    confidence=0.7,
                    keywords_found=["PAGE BREAK"],
                    patient_index=0
                ))
        
        # Method 2: Look for repeating structural patterns
        structured_boundaries = self._detect_structural_repetition(text)
        enhanced_boundaries.extend(structured_boundaries)
        
        # Method 3: Check for demographic block repetition
        demographic_boundaries = self._detect_demographic_blocks(text)
        enhanced_boundaries.extend(demographic_boundaries)
        
        # Merge with initial boundaries
        all_boundaries = initial_boundaries + enhanced_boundaries
        
        # Validate, deduplicate, and sort
        return self._validate_and_deduplicate_boundaries(all_boundaries, text)
    
    def _contexts_suggest_different_patients(self, context_before: str, context_after: str) -> bool:
        """
        Check if two text contexts appear to be from different patients.
        
        Args:
            context_before: Text before a potential boundary
            context_after: Text after a potential boundary
            
        Returns:
            True if contexts likely belong to different patients
        """
        # Check for demographic info in both contexts
        demographic_patterns = [
            r"Name:\s*[A-Za-z\s]+",
            r"DOB:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"Patient\s+ID:\s*\w+",
            r"MRN:\s*\w+"
        ]
        
        before_demographics = []
        after_demographics = []
        
        for pattern in demographic_patterns:
            before_matches = list(re.finditer(pattern, context_before, re.IGNORECASE))
            after_matches = list(re.finditer(pattern, context_after, re.IGNORECASE))
            
            before_demographics.extend([m.group() for m in before_matches])
            after_demographics.extend([m.group() for m in after_matches])
        
        # If both have demographic info, check if they're different
        if before_demographics and after_demographics:
            # Extract actual names or values
            before_values = [re.split(r':\s*', d)[1].strip() for d in before_demographics if ':' in d]
            after_values = [re.split(r':\s*', d)[1].strip() for d in after_demographics if ':' in d]
            
            # Check for different values
            for b_val in before_values:
                for a_val in after_values:
                    # If values are sufficiently different, likely different patients
                    if b_val and a_val and difflib.SequenceMatcher(None, b_val, a_val).ratio() < 0.7:
                        return True
        
        # Check for common "new patient" markers
        new_patient_markers = [
            r"Patient\s*Information",
            r"Patient\s*Demographics",
            r"New\s*Patient",
            r"Insurance\s*Information"
        ]
        
        for marker in new_patient_markers:
            if re.search(marker, context_after, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_structural_repetition(self, text: str) -> List[PatientBoundary]:
        """
        Detect boundaries based on repeating document structures.
        
        Args:
            text: Input text
            
        Returns:
            List of detected boundaries
        """
        boundaries = []
        
        # Look for standard form headers that might repeat for each patient
        header_patterns = [
            r"PATIENT\s*INFORMATION",
            r"CLAIM\s*INFORMATION",
            r"INSURANCE\s*INFORMATION",
            r"ENCOUNTER\s*INFORMATION",
            r"VISIT\s*DETAILS"
        ]
        
        for pattern in header_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if len(matches) > 1:  # Multiple occurrences suggest multiple patients
                for match in matches:
                    # Get some context to improve confidence
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    
                    # Check if context suggests this is a true patient boundary
                    if any(re.search(p, context, re.IGNORECASE) for p in self.keyword_patterns):
                        confidence = 0.85
                    else:
                        confidence = 0.7
                    
                    boundaries.append(PatientBoundary(
                        start_position=match.start(),
                        end_position=None,
                        boundary_type=BoundaryType.FORM_STRUCTURE,
                        confidence=confidence,
                        keywords_found=[match.group()],
                        patient_index=0
                    ))
        
        return boundaries
    
    def _detect_demographic_blocks(self, text: str) -> List[PatientBoundary]:
        """
        Detect patient demographic information blocks.
        
        Args:
            text: Input text
            
        Returns:
            List of detected boundaries
        """
        boundaries = []
        
        # Look for blocks of demographic information
        # These typically contain multiple demographic fields in close proximity
        demographic_fields = [
            r"Name:\s*[A-Za-z\s.,'-]+",
            r"DOB:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"Sex:\s*[MF]",
            r"Gender:\s*[MF]",
            r"Address:\s*[A-Za-z0-9\s.,'-]+",
            r"Phone:\s*[\d()-]+",
            r"MRN:\s*\w+",
            r"Account\s*#:\s*\w+",
            r"SSN:\s*[\dX-]+"
        ]
        
        # Find positions of all demographic fields
        field_positions = []
        for pattern in demographic_fields:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                field_positions.append((match.start(), match.end(), match.group()))
        
        # Sort by position
        field_positions.sort()
        
        # Group fields that are close together
        if field_positions:
            groups = []
            current_group = [field_positions[0]]
            
            for i in range(1, len(field_positions)):
                current_pos = field_positions[i]
                prev_pos = field_positions[i-1]
                
                # If fields are within 200 characters, consider them part of the same group
                if current_pos[0] - prev_pos[1] < 200:
                    current_group.append(current_pos)
                else:
                    # Start a new group
                    if len(current_group) >= 3:  # Only consider groups with at least 3 demographic fields
                        groups.append(current_group)
                    current_group = [current_pos]
            
            # Add the last group if it has enough fields
            if len(current_group) >= 3:
                groups.append(current_group)
            
            # Create boundaries at the start of each demographic group
            for group in groups:
                start_pos = group[0][0]
                keywords = [field[2] for field in group[:3]]  # Use first 3 fields as keywords
                
                boundaries.append(PatientBoundary(
                    start_position=start_pos,
                    end_position=None,
                    boundary_type=BoundaryType.TEXT_BLOCK,
                    confidence=0.8,
                    keywords_found=keywords,
                    patient_index=0
                ))
        
        return boundaries
    
    def _validate_and_deduplicate_boundaries(self, boundaries: List[PatientBoundary], text: str) -> List[PatientBoundary]:
        """
        Validate and deduplicate boundaries.
        
        Args:
            boundaries: List of boundaries to validate
            text: Original text
            
        Returns:
            Validated and deduplicated boundaries
        """
        if not boundaries:
            return []
        
        # Sort by position
        boundaries.sort(key=lambda b: b.start_position)
        
        # Remove duplicates (boundaries that are too close together)
        deduplicated = [boundaries[0]]
        
        for i in range(1, len(boundaries)):
            current = boundaries[i]
            previous = deduplicated[-1]
            
            # If boundaries are more than 100 characters apart, keep both
            if current.start_position - previous.start_position > 100:
                deduplicated.append(current)
            else:
                # Keep the one with higher confidence
                if current.confidence > previous.confidence:
                    deduplicated[-1] = current
        
        # Validate that boundaries make sense
        validated = []
        for boundary in deduplicated:
            # Ensure boundary doesn't split in the middle of a structured field
            context_start = max(0, boundary.start_position - 50)
            context_end = min(len(text), boundary.start_position + 50)
            context = text[context_start:context_end]
            
            # Skip boundaries that split in the middle of typical structured fields
            if re.search(r":\s*$", context[:boundary.start_position - context_start]) and \
               re.match(r"^\s*[A-Za-z0-9]", context[boundary.start_position - context_start:]):
                continue
            
            validated.append(boundary)
        
        # Make sure we don't exceed maximum patients
        if len(validated) > self.max_patients:
            # Keep boundaries with highest confidence
            validated.sort(key=lambda b: b.confidence, reverse=True)
            validated = validated[:self.max_patients]
            # Resort by position
            validated.sort(key=lambda b: b.start_position)
        
        # Assign patient indices
        for i, boundary in enumerate(validated):
            boundary.patient_index = i
        
        return validated


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
        extractor_func,
        page_number: int = 1,
        total_pages: int = 1
    ) -> List[PatientData]:
        """
        Process a multi-patient document.
        
        Args:
            text: Input text containing multiple patients
            extractor_func: Function to extract patient data from text segment
            page_number: Current page number (for multi-page documents)
            total_pages: Total number of pages
            
        Returns:
            List of extracted patient data
        """
        self.logger.info(f"Processing multi-patient document (page {page_number}/{total_pages})")
        
        # Segment text by patient
        segments = self.segmenter.segment_text(text)
        
        # Extract data for each patient segment
        patient_data_list = []
        
        for segment in segments:
            self.logger.debug(f"Processing patient segment {segment.patient_index} on page {page_number}")
            
            try:
                # Extract patient data
                patient_data = await extractor_func(segment.text, segment.patient_index)
                
                if patient_data:
                    # Set page-related metadata
                    patient_data.page_number = page_number
                    
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
                self.logger.error(f"Failed to extract data for patient {segment.patient_index} on page {page_number}: {e}")
        
        # Check for duplicates within this page
        duplicates = self.validator.detect_duplicates(patient_data_list)
        
        if duplicates:
            self.logger.warning(f"Detected {len(duplicates)} potential duplicate patients on page {page_number}")
            patient_data_list = self._merge_duplicates(patient_data_list, duplicates)
        
        self.logger.info(f"Successfully processed {len(patient_data_list)} patients on page {page_number}")
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
