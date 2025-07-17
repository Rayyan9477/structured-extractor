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
            return self._refine_existing_boundaries(text, initial_boundaries)
        
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
