"""
PII Detection Module
Detects Personal Identification Information in uploaded datasets.
Includes Australian Privacy Act compliance (TFN, Medicare, ABN, ACN).
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


# PII Column Name Patterns
PII_COLUMN_PATTERNS = {
    'email': ['email', 'e-mail', 'e_mail', 'mail', 'email_address', 'emailaddress'],
    'phone': ['phone', 'mobile', 'telephone', 'tel', 'cell', 'phone_number', 'mobile_number',
              'contact', 'contact_number'],
    'name': ['name', 'first_name', 'last_name', 'full_name', 'firstname', 'lastname',
             'fname', 'lname', 'given_name', 'surname', 'fullname'],
    'address': ['address', 'street', 'zip', 'postal', 'postcode', 'suburb', 'city',
                'state', 'country', 'location', 'street_address'],
    'id_number': ['ssn', 'tax_file_number', 'tfn', 'medicare', 'passport',
                  'driver_license', 'drivers_license', 'licence', 'abn', 'acn',
                  'social_security', 'national_id', 'citizen_id', 'centrelink'],
    'dob': ['dob', 'birth_date', 'date_of_birth', 'birthdate', 'birthday', 'birth_day'],
    'credit_card': ['credit_card', 'creditcard', 'cc_number', 'card_number', 'pan'],
    'ip_address': ['ip', 'ip_address', 'ipaddress', 'ip_addr'],
    'biometric': ['fingerprint', 'retina', 'face_id', 'facial', 'biometric'],
}

# Content Pattern Regexes
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN_AU = re.compile(r'(\+?61|0)[2-478]\d{8}|\(\d{2}\)\s?\d{4}\s?\d{4}')
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# Australian Privacy Act Specific Patterns
TFN_PATTERN = re.compile(r'\b\d{8,9}\b')  # Tax File Number (8-9 digits)
MEDICARE_PATTERN = re.compile(r'\b\d{10}\b')  # Medicare Number (10 digits)
ABN_PATTERN = re.compile(r'\b\d{11}\b')  # Australian Business Number (11 digits)
ACN_PATTERN = re.compile(r'\b\d{9}\b')  # Australian Company Number (9 digits)


def detect_pii_columns(df: pd.DataFrame) -> Dict:
    """
    Detect columns that may contain PII.

    Args:
        df: DataFrame to analyze

    Returns:
        dict: {
            'suspected_pii_columns': list of column names,
            'pii_types': dict mapping column to suspected PII type,
            'confidence': dict mapping column to confidence level (high/medium/low)
        }

    Detection rules:
    - Email: columns containing '@' symbols, named 'email', 'e-mail', etc.
    - Phone: columns with patterns like (XXX) XXX-XXXX, named 'phone', 'mobile', etc.
    - Names: columns named 'name', 'first_name', 'last_name', 'full_name', etc.
    - Address: columns named 'address', 'street', 'zip', 'postal', 'suburb', etc.
    - ID Numbers: columns named 'ssn', 'tax_file_number', 'tfn', 'medicare', etc.
    - Date of Birth: columns named 'dob', 'birth_date', 'date_of_birth'
    - Credit Card: patterns like XXXX-XXXX-XXXX-XXXX
    - IP Address: patterns like XXX.XXX.XXX.XXX
    - Biometric: columns named 'fingerprint', 'retina', 'face_id'
    """
    suspected_pii_columns = []
    pii_types = {}
    confidence = {}

    for column in df.columns:
        column_lower = column.lower().strip()
        detected_type = None
        conf_level = None

        # Check column name against patterns
        for pii_type, patterns in PII_COLUMN_PATTERNS.items():
            if any(pattern in column_lower for pattern in patterns):
                detected_type = pii_type
                conf_level = 'high'
                break

        # If name-based detection didn't work, check content patterns
        if detected_type is None and df[column].dtype == 'object':
            # Sample non-null values for content analysis
            sample_values = df[column].dropna().astype(str).head(100)

            if len(sample_values) > 0:
                # Check for email patterns
                email_matches = sum(1 for val in sample_values if EMAIL_PATTERN.search(val))
                if email_matches > len(sample_values) * 0.5:  # More than 50% match
                    detected_type = 'email'
                    conf_level = 'high' if email_matches > len(sample_values) * 0.8 else 'medium'

                # Check for phone patterns (Australian format)
                elif any(PHONE_PATTERN_AU.search(val) for val in sample_values):
                    phone_matches = sum(1 for val in sample_values if PHONE_PATTERN_AU.search(val))
                    if phone_matches > len(sample_values) * 0.3:
                        detected_type = 'phone'
                        conf_level = 'high' if phone_matches > len(sample_values) * 0.7 else 'medium'

                # Check for credit card patterns
                elif any(CREDIT_CARD_PATTERN.search(val) for val in sample_values):
                    cc_matches = sum(1 for val in sample_values if CREDIT_CARD_PATTERN.search(val))
                    if cc_matches > len(sample_values) * 0.3:
                        detected_type = 'credit_card'
                        conf_level = 'high' if cc_matches > len(sample_values) * 0.7 else 'medium'

                # Check for IP address patterns
                elif any(IP_PATTERN.search(val) for val in sample_values):
                    ip_matches = sum(1 for val in sample_values if IP_PATTERN.search(val))
                    if ip_matches > len(sample_values) * 0.3:
                        detected_type = 'ip_address'
                        conf_level = 'medium'

                # Check for Australian IDs (TFN, Medicare, ABN, ACN)
                # These are numeric patterns and need careful checking
                elif any(TFN_PATTERN.fullmatch(val.strip()) for val in sample_values if val.strip().isdigit()):
                    tfn_matches = sum(1 for val in sample_values
                                     if val.strip().isdigit() and TFN_PATTERN.fullmatch(val.strip()))
                    if tfn_matches > len(sample_values) * 0.3:
                        detected_type = 'id_number (possible TFN/Medicare/ABN/ACN)'
                        conf_level = 'medium'

        # Check for date of birth (datetime columns with name pattern or old dates)
        if detected_type is None and pd.api.types.is_datetime64_any_dtype(df[column]):
            if any(pattern in column_lower for pattern in PII_COLUMN_PATTERNS['dob']):
                detected_type = 'dob'
                conf_level = 'high'
            else:
                # Check if dates are mostly in the past (birth dates)
                sample_dates = df[column].dropna().head(100)
                if len(sample_dates) > 0:
                    current_year = pd.Timestamp.now().year
                    old_dates = sum(1 for date in sample_dates if date.year < current_year - 10)
                    if old_dates > len(sample_dates) * 0.7:
                        detected_type = 'dob (suspected)'
                        conf_level = 'low'

        # If PII detected, add to results
        if detected_type:
            suspected_pii_columns.append(column)
            pii_types[column] = detected_type
            confidence[column] = conf_level

    return {
        'suspected_pii_columns': suspected_pii_columns,
        'pii_types': pii_types,
        'confidence': confidence
    }


def get_pii_statistics(df: pd.DataFrame, pii_columns: List[str]) -> Dict:
    """
    Get statistics about PII columns.

    Args:
        df: DataFrame
        pii_columns: List of column names identified as PII

    Returns:
        dict: Statistics for each PII column including:
        - Number of non-null values
        - Percentage of rows containing PII
        - Sample values (masked for display)
    """
    stats = {}

    for column in pii_columns:
        if column in df.columns:
            non_null_count = df[column].notna().sum()
            total_count = len(df)
            percentage = (non_null_count / total_count * 100) if total_count > 0 else 0

            # Get sample values (masked)
            sample_values = df[column].dropna().head(3).tolist()

            stats[column] = {
                'non_null_count': non_null_count,
                'percentage': percentage,
                'sample_values_masked': [mask_pii_for_display(str(val), 'generic')
                                        for val in sample_values]
            }

    return stats


def mask_pii_for_display(value: str, pii_type: str) -> str:
    """
    Mask PII values for safe display.

    Args:
        value: The PII value to mask
        pii_type: Type of PII (email, phone, etc.)

    Returns:
        str: Masked value

    Examples:
        'john.doe@email.com' -> 'j***@e***'
        '0412345678' -> '04****5678'
        'John Smith' -> 'J*** S***'
    """
    if not value or len(value) == 0:
        return value

    # Email masking
    if pii_type == 'email' or '@' in value:
        if '@' in value:
            local, domain = value.split('@', 1)
            if len(local) > 2:
                local_masked = local[0] + '*' * min(3, len(local) - 1)
            else:
                local_masked = local[0] + '*'

            if '.' in domain:
                domain_parts = domain.split('.')
                domain_masked = domain_parts[0][0] + '*' * min(3, len(domain_parts[0]) - 1)
                if len(domain_parts) > 1:
                    domain_masked += '.' + domain_parts[-1]
            else:
                domain_masked = domain[0] + '*' * min(3, len(domain) - 1)

            return f"{local_masked}@{domain_masked}"

    # Phone masking (keep first 2 and last 4 digits)
    elif pii_type == 'phone':
        # Remove non-digits for masking
        digits_only = re.sub(r'\D', '', value)
        if len(digits_only) >= 6:
            return digits_only[:2] + '*' * (len(digits_only) - 6) + digits_only[-4:]
        else:
            return digits_only[0] + '*' * (len(digits_only) - 1) if len(digits_only) > 0 else value

    # Name masking (first letter + ***)
    elif pii_type == 'name':
        parts = value.split()
        masked_parts = []
        for part in parts:
            if len(part) > 1:
                masked_parts.append(part[0] + '*' * min(3, len(part) - 1))
            else:
                masked_parts.append(part)
        return ' '.join(masked_parts)

    # Generic masking (show first and last character)
    else:
        if len(value) <= 2:
            return '*' * len(value)
        elif len(value) <= 4:
            return value[0] + '*' * (len(value) - 2) + value[-1]
        else:
            return value[0] + '*' * min(6, len(value) - 2) + value[-1]
