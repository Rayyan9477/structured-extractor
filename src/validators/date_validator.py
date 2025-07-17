"""
Date Validation and Standardization Module

Provides comprehensive date validation, standardization, and consistency checking
for medical documentation dates.
"""

import re
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import dateparser
from dateutil import parser

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class DateValidator:
    """Validates and standardizes date formats in medical documentation."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize date validator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get date validation settings
        self.date_config = config.get("validation", {}).get("dates", {})
        self.min_year = self.date_config.get("min_year", 1900)
        self.max_year = self.date_config.get("max_year", datetime.now().year + 1)
        self.us_date_format = self.date_config.get("us_date_format", True)
        
        # Common date patterns
        self.date_patterns = [
            # MM/DD/YYYY
            r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b',
            # YYYY/MM/DD
            r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])\b',
            # DD/MM/YYYY (less common in US medical records)
            r'\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b',
            # Month DD, YYYY
            r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(19|20)\d{2}\b',
        ]
    
    def standardize_date(self, date_str: str) -> Optional[str]:
        """
        Standardize date string to YYYY-MM-DD format.
        
        Args:
            date_str: Input date string
            
        Returns:
            Standardized date string or None if invalid
        """
        if not date_str:
            return None
        
        # Try to parse the date
        try:
            # Use dateparser for flexible parsing
            parsed_date = dateparser.parse(
                date_str, 
                settings={
                    'DATE_ORDER': 'MDY' if self.us_date_format else 'DMY',
                    'STRICT_PARSING': False
                }
            )
            
            if not parsed_date:
                # Fallback to dateutil parser
                parsed_date = parser.parse(date_str, dayfirst=not self.us_date_format)
            
            # Validate year range
            if parsed_date.year < self.min_year or parsed_date.year > self.max_year:
                self.logger.warning(f"Date out of valid range: {date_str}")
                return None
            
            # Return standardized format
            return parsed_date.strftime('%Y-%m-%d')
            
        except (ValueError, parser.ParserError) as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def validate_date_consistency(
        self, 
        dates: Dict[str, str],
        reference_date: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Validate consistency among multiple dates.
        
        Args:
            dates: Dictionary of date fields and their values
            reference_date: Optional reference date (e.g., today)
            
        Returns:
            Dictionary with date fields and their validity
        """
        if not dates:
            return {}
        
        # Standardize all dates
        standardized_dates = {}
        for field, date_str in dates.items():
            std_date = self.standardize_date(date_str)
            if std_date:
                standardized_dates[field] = std_date
        
        # Set reference date if not provided
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert to datetime objects
        date_objects = {}
        for field, date_str in standardized_dates.items():
            date_objects[field] = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        ref_date_obj = datetime.strptime(reference_date, '%Y-%m-%d').date()
        
        # Common date consistency rules for medical documentation
        validity = {}
        
        # Rule 1: Date of birth should be in the past
        if 'date_of_birth' in date_objects:
            validity['date_of_birth'] = date_objects['date_of_birth'] < ref_date_obj
        
        # Rule 2: Date of service should be in the past or today
        if 'date_of_service' in date_objects:
            validity['date_of_service'] = date_objects['date_of_service'] <= ref_date_obj
        
        # Rule 3: Claim date should be on or after date of service
        if 'claim_date' in date_objects and 'date_of_service' in date_objects:
            validity['claim_date'] = date_objects['claim_date'] >= date_objects['date_of_service']
        
        # Rule 4: No future dates (except for scheduled appointments)
        for field, date_obj in date_objects.items():
            if field not in validity and field != 'next_appointment':
                validity[field] = date_obj <= ref_date_obj
        
        # Rule 5: Date of birth should be reasonable (not too old)
        if 'date_of_birth' in date_objects:
            max_age = 120  # Maximum reasonable age
            min_date = ref_date_obj - timedelta(days=365.25 * max_age)
            validity['date_of_birth'] = validity.get('date_of_birth', True) and date_objects['date_of_birth'] > min_date
        
        return validity
    
    def extract_date_from_text(self, text: str) -> List[str]:
        """
        Extract dates from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted date strings
        """
        dates = []
        
        # Search for dates using patterns
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group()
                standardized = self.standardize_date(date_str)
                if standardized and standardized not in dates:
                    dates.append(standardized)
        
        return dates
    
    def is_valid_age(self, age: Union[int, str], reference_date: Optional[str] = None) -> bool:
        """
        Validate if age is reasonable.
        
        Args:
            age: Age to validate
            reference_date: Optional reference date
            
        Returns:
            True if age is valid
        """
        try:
            # Convert to int if string
            if isinstance(age, str):
                age = int(age.strip())
            
            # Check range (0-120 years is reasonable for medical records)
            return 0 <= age <= 120
            
        except ValueError:
            return False
