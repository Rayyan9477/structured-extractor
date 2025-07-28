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
            "PATIENT INFORMATION", "DEMOGRAPHICS", "DATE OF BIRTH",
            "CPT CODE", "CPT CODES", "PROCEDURE CODE", "PROCEDURE CODES",
            "ICD", "ICD-10", "ICD10", "DIAGNOSIS CODE", "DIAGNOSIS CODES",
            "PRIMARY DIAGNOSIS", "SECONDARY DIAGNOSIS", "BILLING CODE"
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
        
        # Enhanced CPT and ICD-10 code detection patterns
        self.cpt_code_patterns = [
            r"(?:CPT|PROCEDURE)[\s#:]*(\d{5})",
            r"\b(\d{5})\b(?=.*(?:CPT|PROCEDURE|BILLING))",
            r"(?:CODE|PROC)[\s#:]*(\d{5})",
            r"\b(\d{5})\s*[-–]\s*[A-Za-z]"  # CPT followed by description
        ]
        
        self.icd10_code_patterns = [
            r"(?:ICD|DIAGNOSIS)[\s#:-]*([A-Z]\d{2}(?:\.[A-Z0-9]{1,3})?)",
            r"\b([A-Z]\d{2}(?:\.[A-Z0-9]{1,3})?)\b(?=.*(?:ICD|DIAGNOSIS))",
            r"(?:DX|DIAG)[\s#:]*([A-Z]\d{2}(?:\.[A-Z0-9]{1,3})?)",
            r"\b([A-Z]\d{2}(?:\.[A-Z0-9]{1,3})?)\s*[-–]\s*[A-Za-z]"  # ICD followed by description
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
        
        # Method 6: CPT/ICD code grouping detection (NEW)
        boundaries.extend(self._detect_code_grouping_boundaries(text))
        
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
        # Enhanced demographic indicators for better patient detection
        demographic_indicators = [
            r"\bDOB\b", r"\bBIRTH\b", r"\bAGE\b", r"\bSEX\b", r"\bGENDER\b",
            r"\bADDRESS\b", r"\bPHONE\b", r"\bEMAIL\b", r"\bINSURANCE\b",
            r"\bPATIENT\s+ID\b", r"\bMEDICAL\s+RECORD\b", r"\bMRN\b",
            r"\bACCOUNT\s+NUMBER\b", r"\bSSN\b", r"\bSOCIAL\s+SECURITY\b",
            r"\bDATE\s+OF\s+SERVICE\b", r"\bSERVICE\s+DATE\b", r"\bDOS\b"
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
    
    def _detect_code_grouping_boundaries(self, text: str) -> List[PatientBoundary]:
        """
        Detect patient boundaries based on CPT and ICD-10 code groupings.
        
        This method identifies potential patient boundaries by analyzing where
        clusters of medical codes appear, as different patients typically have
        distinct sets of codes.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected boundaries based on code patterns
        """
        boundaries = []
        
        # Find all CPT and ICD codes with their positions
        code_positions = []
        
        # Compile patterns if not already done
        cpt_patterns = [re.compile(p, re.IGNORECASE) for p in self.cpt_code_patterns]
        icd_patterns = [re.compile(p, re.IGNORECASE) for p in self.icd10_code_patterns]
        
        # Find CPT codes
        for pattern in cpt_patterns:
            for match in pattern.finditer(text):
                code_positions.append({
                    'type': 'CPT',
                    'code': match.group(1) if match.groups() else match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'match_text': match.group(0)
                })
        
        # Find ICD-10 codes  
        for pattern in icd_patterns:
            for match in pattern.finditer(text):
                code_positions.append({
                    'type': 'ICD10',
                    'code': match.group(1) if match.groups() else match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'match_text': match.group(0)
                })
        
        # Sort by position
        code_positions.sort(key=lambda x: x['start'])
        
        if len(code_positions) < 4:  # Need at least 4 codes to detect groupings
            return boundaries
        
        # Analyze code clustering
        clusters = self._cluster_codes_by_proximity(code_positions)
        
        # Create boundaries between distinct clusters
        for i in range(1, len(clusters)):
            prev_cluster = clusters[i-1]
            current_cluster = clusters[i]
            
            # Calculate distance between clusters
            cluster_gap = current_cluster[0]['start'] - prev_cluster[-1]['end']
            
            # If clusters are sufficiently separated, create a boundary
            if cluster_gap > 200:  # Minimum gap between patient sections
                # Look for additional context that suggests a new patient
                gap_text = text[prev_cluster[-1]['end']:current_cluster[0]['start']]
                
                # Check if gap contains patient-related keywords
                patient_keywords_in_gap = sum(1 for kw in self.separation_keywords 
                                            if kw.lower() in gap_text.lower())
                
                confidence = 0.7  # Base confidence for code clustering
                if patient_keywords_in_gap > 0:
                    confidence += 0.2
                if cluster_gap > 500:  # Large gap increases confidence
                    confidence += 0.1
                
                # Find the best position for the boundary (start of new cluster)
                boundary_position = current_cluster[0]['start']
                
                # Look backward for a good boundary marker
                search_start = max(0, boundary_position - 100)
                boundary_context = text[search_start:boundary_position + 50]
                
                # Find line breaks or other natural boundaries
                lines = boundary_context.split('\n')
                if len(lines) > 1:
                    # Prefer to place boundary at start of line containing first code
                    for j, line in enumerate(lines):
                        if current_cluster[0]['match_text'] in line:
                            line_start = search_start + sum(len(lines[k]) + 1 for k in range(j))
                            boundary_position = line_start
                            break
                
                boundaries.append(PatientBoundary(
                    start_position=boundary_position,
                    end_position=None,
                    boundary_type=BoundaryType.PATTERN_REPEAT,
                    confidence=min(confidence, 1.0),
                    keywords_found=[f"Code cluster: {len(current_cluster)} codes"],
                    patient_index=0  # Will be set later
                ))
        
        return boundaries
    
    def _cluster_codes_by_proximity(self, code_positions: List[Dict]) -> List[List[Dict]]:
        """
        Cluster codes by their proximity in the text.
        
        Args:
            code_positions: List of code position dictionaries
            
        Returns:
            List of code clusters
        """
        if not code_positions:
            return []
        
        clusters = [[code_positions[0]]]
        
        for i in range(1, len(code_positions)):
            current_code = code_positions[i]
            last_cluster = clusters[-1]
            last_code_in_cluster = last_cluster[-1]
            
            # If codes are close together (within 300 characters), add to same cluster
            if current_code['start'] - last_code_in_cluster['end'] <= 300:
                last_cluster.append(current_code)
            else:
                # Start a new cluster
                clusters.append([current_code])
        
        # Filter out clusters that are too small (less than 2 codes)
        clusters = [cluster for cluster in clusters if len(cluster) >= 2]
        
        return clusters
    
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
            try:
                # Convert string date to date object for comparison
                if isinstance(patient_data.date_of_birth, str):
                    from datetime import datetime
                    dob_date = datetime.strptime(patient_data.date_of_birth, '%Y-%m-%d').date()
                else:
                    dob_date = patient_data.date_of_birth
                
                if dob_date > date.today():
                    errors.append("Date of birth is in the future")
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid date format for date of birth: {patient_data.date_of_birth}")
        
        # Check data consistency
        if hasattr(patient_data, 'service_info') and patient_data.service_info and hasattr(patient_data.service_info, 'date_of_service') and patient_data.service_info.date_of_service:
            if patient_data.date_of_birth:
                try:
                    # Convert string dates to date objects for comparison
                    from datetime import datetime
                    
                    if isinstance(patient_data.date_of_birth, str):
                        dob_date = datetime.strptime(patient_data.date_of_birth, '%Y-%m-%d').date()
                    else:
                        dob_date = patient_data.date_of_birth
                    
                    if isinstance(patient_data.service_info.date_of_service, str):
                        service_date = datetime.strptime(patient_data.service_info.date_of_service, '%Y-%m-%d').date()
                    else:
                        service_date = patient_data.service_info.date_of_service
                    
                    # Check if service date is reasonable relative to birth date
                    age_at_service = (service_date - dob_date).days / 365.25
                    if age_at_service < 0:
                        errors.append("Service date is before birth date")
                    elif age_at_service > 150:
                        errors.append("Patient age at service is unrealistic")
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid date format for comparison: {e}")
        
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
                    patient_data.total_pages = total_pages
                    
                    # Store the text segment used for extraction
                    patient_data.text_segment = segment.text
                    
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
    
    async def process_multi_page_document(
        self,
        page_texts: List[str],
        extractor_func,
    ) -> List[PatientData]:
        """
        Process a multi-page document with patient detection across pages.
        
        Args:
            page_texts: List of OCR text for each page
            extractor_func: Function to extract patient data from text segment
            
        Returns:
            List of extracted patient data with cross-page merging
        """
        self.logger.info(f"Processing multi-page document with {len(page_texts)} pages")
        
        all_patients = []
        
        # First pass: process each page independently
        for page_num, page_text in enumerate(page_texts, 1):
            page_patients = await self.process_multi_patient_document(
                page_text,
                extractor_func,
                page_number=page_num,
                total_pages=len(page_texts)
            )
            
            # Add page patients to the full list
            all_patients.extend(page_patients)
            
            # Force a short delay to prevent CPU overload
            if page_num % 5 == 0:
                await asyncio.sleep(0.1)
        
        # Enhanced second pass: use the improved patient detection across pages
        optimized_patients = self.optimize_patient_detection(all_patients)
        
        # Log the changes in patient count
        if len(optimized_patients) != len(all_patients):
            self.logger.info(f"Patient optimization: reduced from {len(all_patients)} to {len(optimized_patients)} patients")
        
        # Return the optimized patient list
        return optimized_patients
    
    def detect_cross_page_patients(self, patients: List[PatientData]) -> List[Tuple[int, int, float]]:
        """
        Detect patients that appear across multiple pages.
        
        Args:
            patients: List of all extracted patients
            
        Returns:
            List of cross-page duplicate pairs (index1, index2, similarity)
        """
        cross_page_duplicates = []
        
        # Group patients by page
        patients_by_page = {}
        for i, patient in enumerate(patients):
            page = patient.page_number
            if page not in patients_by_page:
                patients_by_page[page] = []
            patients_by_page[page].append((i, patient))
        
        # Compare patients across adjacent pages
        for page in sorted(patients_by_page.keys()):
            # Skip the last page
            if page + 1 not in patients_by_page:
                continue
                
            current_page_patients = patients_by_page[page]
            next_page_patients = patients_by_page[page + 1]
            
            # Compare each patient on current page with each on next page
            for i_idx, i_patient in current_page_patients:
                for j_idx, j_patient in next_page_patients:
                    similarity = self.validator._calculate_patient_similarity(i_patient, j_patient)
                    
                    # Use a lower threshold for cross-page matching
                    if similarity > 0.7:  # Slightly lower than within-page threshold
                        cross_page_duplicates.append((i_idx, j_idx, similarity))
                        
                        self.logger.debug(
                            f"Cross-page match between page {page}:{i_patient.first_name} {i_patient.last_name} "
                            f"and page {page+1}:{j_patient.first_name} {j_patient.last_name} "
                            f"(similarity: {similarity:.2f})"
                        )
        
        return cross_page_duplicates

    def optimize_patient_detection(self, patients: List[PatientData]) -> List[PatientData]:
        """
        Optimize patient detection across pages using additional heuristics.
        
        This method enhances the cross-page patient detection by applying 
        additional heuristics beyond simple similarity matching:
        
        1. Checks for matching CPT/ICD codes across pages
        2. Identifies continuation markers in text
        3. Analyzes text patterns that typically indicate continued data
        
        Args:
            patients: List of all extracted patients
            
        Returns:
            Optimized list of patients with improved cross-page relationships
        """
        self.logger.info(f"Optimizing patient detection across pages for {len(patients)} patients")
        
        # First apply the regular cross-page detection
        cross_page_matches = self.detect_cross_page_patients(patients)
        
        # Find additional matches based on CPT/ICD code similarity
        code_based_matches = self._find_code_based_matches(patients)
        
        # Combine all matches, avoiding duplicates
        all_matches = cross_page_matches.copy()
        for i, j, sim in code_based_matches:
            # Check if this pair is already matched
            if not any(i == x and j == y for x, y, _ in cross_page_matches):
                all_matches.append((i, j, sim))
        
        # If we found more matches, merge them
        if len(all_matches) > len(cross_page_matches):
            self.logger.info(f"Found {len(all_matches) - len(cross_page_matches)} additional cross-page matches")
            merged_patients = self._merge_duplicates(patients, all_matches)
            
            # Update multi-page flags and page numbers
            for patient in merged_patients:
                if len(patient.page_numbers) > 1:
                    patient.spans_multiple_pages = True
                    
            # Sort by page number
            merged_patients.sort(key=lambda p: (p.page_number, getattr(p, 'patient_index', 0)))
            return merged_patients
        
        # If no additional matches found, return the original list
        return patients
    
    def _find_code_based_matches(self, patients: List[PatientData]) -> List[Tuple[int, int, float]]:
        """
        Find patient matches based on CPT and ICD code similarity.
        
        Args:
            patients: List of all patients
            
        Returns:
            List of matches (index1, index2, similarity)
        """
        matches = []
        
        # Group patients by page
        patients_by_page = {}
        for i, patient in enumerate(patients):
            page = patient.page_number
            if page not in patients_by_page:
                patients_by_page[page] = []
            patients_by_page[page].append((i, patient))
        
        # Compare patients on adjacent pages
        for page in sorted(patients_by_page.keys()):
            if page + 1 not in patients_by_page:
                continue
                
            current_page_patients = patients_by_page[page]
            next_page_patients = patients_by_page[page + 1]
            
            for i_idx, i_patient in current_page_patients:
                for j_idx, j_patient in next_page_patients:
                    # Skip if no codes to compare
                    if not i_patient.cpt_codes and not i_patient.icd10_codes:
                        continue
                        
                    # Calculate code similarity
                    code_similarity = self._calculate_code_similarity(i_patient, j_patient)
                    
                    # Use a moderate threshold for code-based matching
                    if code_similarity > 0.6:
                        self.logger.debug(
                            f"Code-based match between page {page}:{i_patient.first_name} {i_patient.last_name} "
                            f"and page {page+1}:{j_patient.first_name} {j_patient.last_name} "
                            f"(similarity: {code_similarity:.2f})"
                        )
                        matches.append((i_idx, j_idx, code_similarity))
        
        return matches
    
    def _calculate_code_similarity(self, patient1: PatientData, patient2: PatientData) -> float:
        """
        Calculate similarity between two patients based on their CPT and ICD codes.
        
        Args:
            patient1: First patient
            patient2: Second patient
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get sets of codes from both patients
        cpt1 = {c.code for c in patient1.cpt_codes}
        cpt2 = {c.code for c in patient2.cpt_codes}
        icd1 = {c.code for c in patient1.icd10_codes}
        icd2 = {c.code for c in patient2.icd10_codes}
        
        # Calculate Jaccard similarity for CPT codes
        cpt_sim = 0.0
        if cpt1 or cpt2:
            intersection = len(cpt1.intersection(cpt2))
            union = len(cpt1.union(cpt2))
            cpt_sim = intersection / union if union > 0 else 0.0
        
        # Calculate Jaccard similarity for ICD codes
        icd_sim = 0.0
        if icd1 or icd2:
            intersection = len(icd1.intersection(icd2))
            union = len(icd1.union(icd2))
            icd_sim = intersection / union if union > 0 else 0.0
        
        # Combine similarities, weighting CPT codes slightly higher
        if cpt1 or cpt2 or icd1 or icd2:
            combined_sim = (0.6 * cpt_sim + 0.4 * icd_sim) if (cpt1 or cpt2) and (icd1 or icd2) else max(cpt_sim, icd_sim)
            return combined_sim
        
        return 0.0
    
    def _merge_duplicates(
        self, 
        patients: List[PatientData], 
        duplicates: List[Tuple[int, int, float]]
    ) -> List[PatientData]:
        """
        Merge duplicate patient records.
        
        Args:
            patients: List of patient data records
            duplicates: List of duplicate pairs (index1, index2, similarity)
            
        Returns:
            List of merged patient records
        """
        self.logger.debug(f"Merging {len(duplicates)} duplicate patient pairs")
        
        to_remove = set()
        merge_groups = []
        
        # Group duplicates together
        for i, j, similarity in duplicates:
            # Skip if either patient is already marked for removal
            if i in to_remove or j in to_remove:
                continue
                
            # Find if either patient is already in a merge group
            found_group = False
            for group in merge_groups:
                if i in group or j in group:
                    group.add(i)
                    group.add(j)
                    found_group = True
                    break
                    
            # If not found in any group, create a new group
            if not found_group:
                merge_groups.append({i, j})
        
        # Process each merge group
        merged_patients = []
        processed_indices = set()
        
        for group in merge_groups:
            # Sort patients in the group by confidence
            group_patients = [(idx, patients[idx]) for idx in group]
            group_patients.sort(key=lambda x: x[1].extraction_confidence, reverse=True)
            
            # Take the highest confidence patient as base
            base_idx, base_patient = group_patients[0]
            processed_indices.add(base_idx)
            
            # Merge information from other patients
            for idx, patient in group_patients[1:]:
                processed_indices.add(idx)
                base_patient = self._combine_patient_data(base_patient, patient)
            
            merged_patients.append(base_patient)
        
        # Add patients that weren't part of any merge group
        for i, patient in enumerate(patients):
            if i not in processed_indices:
                merged_patients.append(patient)
        
        self.logger.debug(f"Reduced from {len(patients)} to {len(merged_patients)} patients after merging")
        return merged_patients
    
    def _combine_patient_data(self, primary: PatientData, secondary: PatientData) -> PatientData:
        """
        Combine information from two patient records, prioritizing the primary record.
        
        Args:
            primary: Primary patient record (higher confidence)
            secondary: Secondary patient record to merge from
            
        Returns:
            Combined patient record
        """
        # Start with primary patient data
        combined = primary
        
        # Fill in missing fields from secondary
        if not primary.first_name and secondary.first_name:
            combined.first_name = secondary.first_name
            
        if not primary.last_name and secondary.last_name:
            combined.last_name = secondary.last_name
            
        if not primary.date_of_birth and secondary.date_of_birth:
            combined.date_of_birth = secondary.date_of_birth
            
        if not primary.patient_id and secondary.patient_id:
            combined.patient_id = secondary.patient_id
            
        # Combine CPT codes (avoid duplicates)
        existing_cpt_codes = {code.code for code in primary.cpt_codes}
        for code in secondary.cpt_codes:
            if code.code not in existing_cpt_codes:
                combined.cpt_codes.append(code)
                existing_cpt_codes.add(code.code)
                
        # Combine ICD-10 codes (avoid duplicates)
        existing_icd_codes = {code.code for code in primary.icd10_codes}
        for code in secondary.icd10_codes:
            if code.code not in existing_icd_codes:
                combined.icd10_codes.append(code)
                existing_icd_codes.add(code.code)
                
        # Set multi-page flag if patients are from different pages
        if primary.page_number != secondary.page_number:
            combined.spans_multiple_pages = True
            combined.page_numbers = sorted(set([primary.page_number, secondary.page_number]))
            
        # Update extraction confidence
        combined.extraction_confidence = max(primary.extraction_confidence, secondary.extraction_confidence)
        
        return combined
