import pandas as pd
import numpy as np
import chardet
import io
from typing import Dict, Any, Tuple, Optional, Union
from modules.data_preprocessor import DataPreprocessor

def detect_encoding(file) -> str:
    """
    Detect the encoding of a file.

    Args:
        file: File-like object or bytes

    Returns:
        str: Detected encoding
    """
    try:
        if hasattr(file, 'read'):
            sample = file.read(10000)
            file.seek(0)
        else:
            sample = file[:10000]

        result = chardet.detect(sample)
        return result['encoding'] if result['encoding'] else 'utf-8'
    except:
        return 'utf-8'

def load_data(file, file_type: str, filename: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from various file formats with error handling.
    Supports CSV, Excel, JSON, Parquet, TSV, TXT, Feather, HDF5, Pickle, SQLite, HTML, XML.

    Args:
        file: Uploaded file object
        file_type: Type of file (e.g., 'csv', 'excel', 'json', etc.)
        filename: Original filename (used for format detection)
        **kwargs: Additional parameters for specific formats

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    metadata = {}
    preprocessor = DataPreprocessor()

    try:
        # Check if preprocessing is needed (non-CSV/Excel formats)
        needs_preprocessing = preprocessor.is_supported_format(file_type.lower()) and file_type.lower() not in ['csv', 'excel', 'xlsx', 'xls']

        if needs_preprocessing:
            # Use preprocessor for other formats
            df, preprocess_metadata = preprocessor.preprocess_file(file, filename or f"file.{file_type}", **kwargs)
            metadata.update(preprocess_metadata)
            metadata['preprocessing_used'] = True
            metadata['encoding'] = 'N/A (Preprocessed)'
            metadata['separator'] = 'N/A (Preprocessed)'

        elif file_type.lower() == 'csv':
            # Original CSV handling
            encoding = detect_encoding(file)
            metadata['encoding'] = encoding
            metadata['preprocessing_used'] = False

            # Try different separators
            separators = [',', ';', '\t', '|']
            df = None

            for sep in separators:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding, separator=sep, low_memory=False)
                    if df.shape[1] > 1:  # Good separator found
                        metadata['separator'] = sep
                        break
                except:
                    continue

            if df is None:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, low_memory=False)
                metadata['separator'] = ','

        elif file_type.lower() in ['excel', 'xlsx', 'xls']:
            # Original Excel handling
            df = pd.read_excel(file, engine='openpyxl' if file_type.lower() in ['excel', 'xlsx'] else 'xlrd')
            metadata['encoding'] = 'N/A (Excel)'
            metadata['separator'] = 'N/A (Excel)'
            metadata['preprocessing_used'] = False
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Get basic metadata
        basic_metadata = get_basic_metadata(df)
        metadata.update(basic_metadata)

        return df, metadata

    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def get_basic_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract basic metadata from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing metadata
    """
    metadata = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': estimate_memory_usage(df),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'total_missing': df.isnull().sum().sum(),
        'total_missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    }

    return metadata

def estimate_memory_usage(df: pd.DataFrame) -> float:
    """
    Estimate memory usage of DataFrame in MB.

    Args:
        df: Input DataFrame

    Returns:
        float: Memory usage in MB
    """
    try:
        memory_usage = df.memory_usage(deep=True).sum()
        return round(memory_usage / (1024 * 1024), 2)
    except:
        return 0.0

def preview_data(df: pd.DataFrame, n_rows: int = 10, from_end: bool = False) -> pd.DataFrame:
    """
    Preview first or last N rows of data.

    Args:
        df: Input DataFrame
        n_rows: Number of rows to show
        from_end: If True, show last N rows

    Returns:
        DataFrame with preview data
    """
    if from_end:
        return df.tail(n_rows)
    else:
        return df.head(n_rows)

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate DataFrame and return validation results.

    Args:
        df: Input DataFrame

    Returns:
        Dict with validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }

    # Check if DataFrame is empty
    if df.empty:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame is empty")
        return validation

    # Check if DataFrame has columns
    if len(df.columns) == 0:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame has no columns")
        return validation

    # Check for very small datasets
    if len(df) < 5:
        validation['warnings'].append("Dataset is very small (< 5 rows)")

    # Check for single column
    if len(df.columns) == 1:
        validation['warnings'].append("Dataset has only one column")

    # Check for all missing values in any column
    all_missing_cols = df.columns[df.isnull().all()].tolist()
    if all_missing_cols:
        validation['warnings'].append(f"Columns with all missing values: {all_missing_cols}")

    # Check memory usage
    memory_mb = estimate_memory_usage(df)
    if memory_mb > 500:
        validation['warnings'].append(f"Large dataset detected ({memory_mb:.1f} MB)")

    return validation

def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about each column.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with column information
    """
    column_info = []

    for col in df.columns:
        info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].notna().sum(),
            'Null Count': df[col].isnull().sum(),
            'Null %': round(df[col].isnull().sum() / len(df) * 100, 2),
            'Unique Values': df[col].nunique(),
            'Unique %': round(df[col].nunique() / len(df) * 100, 2)
        }

        # Add type-specific info
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            info['Min'] = df[col].min() if df[col].notna().any() else None
            info['Max'] = df[col].max() if df[col].notna().any() else None
            info['Mean'] = round(df[col].mean(), 2) if df[col].notna().any() else None
        else:
            info['Min'] = None
            info['Max'] = None
            info['Mean'] = None

        # Most common value
        if df[col].notna().any():
            most_common = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else None
            info['Most Common'] = str(most_common)[:50] if most_common is not None else None
        else:
            info['Most Common'] = None

        column_info.append(info)

    return pd.DataFrame(column_info)