"""
Data Preprocessor Module

Handles various file formats and converts them to pandas DataFrame
for seamless integration with the EDA tool.

Supported Formats:
- JSON (flat, nested, records, lines)
- Parquet
- TSV (Tab-separated values)
- TXT (custom delimiters)
- Feather
- HDF5
- Pickle
- SQLite
- HTML tables
- XML (basic structure)
"""

import pandas as pd
import numpy as np
import json
import io
import sqlite3
import tempfile
import os
from typing import Dict, Any, Tuple, Optional, Union
import chardet
import xml.etree.ElementTree as ET
from pathlib import Path


class DataPreprocessor:
    """Handles conversion of various file formats to pandas DataFrame"""

    SUPPORTED_FORMATS = {
        'json': 'JSON',
        'parquet': 'Parquet',
        'tsv': 'Tab-Separated Values',
        'txt': 'Text File',
        'feather': 'Feather',
        'h5': 'HDF5',
        'hdf5': 'HDF5',
        'pkl': 'Pickle',
        'pickle': 'Pickle',
        'db': 'SQLite Database',
        'sqlite': 'SQLite Database',
        'sqlite3': 'SQLite Database',
        'html': 'HTML Table',
        'htm': 'HTML Table',
        'xml': 'XML'
    }

    def __init__(self):
        self.conversion_log = []

    def detect_format(self, file, filename: str) -> str:
        """
        Detect file format based on extension and content.

        Args:
            file: Uploaded file object
            filename: Name of the file

        Returns:
            str: Detected format
        """
        # Get extension
        ext = Path(filename).suffix.lower().lstrip('.')

        # Check if supported
        if ext in self.SUPPORTED_FORMATS:
            return ext

        # Try to detect from content for ambiguous cases
        if ext in ['txt', '']:
            file.seek(0)
            sample = file.read(1000)
            file.seek(0)

            # Check for tab-separated
            if b'\t' in sample and b',' not in sample[:200]:
                return 'tsv'

            # Check for JSON
            try:
                if isinstance(sample, bytes):
                    sample = sample.decode('utf-8')
                json.loads(sample)
                return 'json'
            except:
                pass

        return ext

    def is_supported_format(self, format_type: str) -> bool:
        """Check if format is supported"""
        return format_type.lower() in self.SUPPORTED_FORMATS

    def get_supported_formats(self) -> Dict[str, str]:
        """Get dictionary of supported formats"""
        return self.SUPPORTED_FORMATS.copy()

    def preprocess_file(self, file, filename: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main preprocessing pipeline - converts file to DataFrame.

        Args:
            file: Uploaded file object
            filename: Name of the file
            **kwargs: Additional parameters for specific converters

        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        metadata = {
            'original_filename': filename,
            'original_format': None,
            'conversion_status': 'pending',
            'warnings': [],
            'errors': []
        }

        try:
            # Detect format
            format_type = self.detect_format(file, filename)
            metadata['original_format'] = format_type

            if not self.is_supported_format(format_type):
                raise ValueError(f"Unsupported file format: {format_type}")

            # Route to appropriate converter
            file.seek(0)

            if format_type == 'json':
                df = self._convert_json(file, **kwargs)
            elif format_type == 'parquet':
                df = self._convert_parquet(file)
            elif format_type == 'tsv':
                df = self._convert_tsv(file)
            elif format_type == 'txt':
                df = self._convert_txt(file, **kwargs)
            elif format_type == 'feather':
                df = self._convert_feather(file)
            elif format_type in ['h5', 'hdf5']:
                df = self._convert_hdf5(file, **kwargs)
            elif format_type in ['pkl', 'pickle']:
                df = self._convert_pickle(file)
            elif format_type in ['db', 'sqlite', 'sqlite3']:
                df = self._convert_sqlite(file, **kwargs)
            elif format_type in ['html', 'htm']:
                df = self._convert_html(file, **kwargs)
            elif format_type == 'xml':
                df = self._convert_xml(file, **kwargs)
            else:
                raise ValueError(f"No converter available for: {format_type}")

            # Validate DataFrame
            if df is None or df.empty:
                raise ValueError("Conversion resulted in empty DataFrame")

            metadata['conversion_status'] = 'success'
            metadata['rows_converted'] = len(df)
            metadata['columns_converted'] = len(df.columns)

            # Add any warnings
            if df.isnull().all().any():
                cols_all_null = df.columns[df.isnull().all()].tolist()
                metadata['warnings'].append(f"Columns with all null values: {cols_all_null}")

            self.conversion_log.append({
                'filename': filename,
                'format': format_type,
                'status': 'success',
                'rows': len(df),
                'columns': len(df.columns)
            })

            return df, metadata

        except Exception as e:
            metadata['conversion_status'] = 'failed'
            metadata['errors'].append(str(e))

            self.conversion_log.append({
                'filename': filename,
                'format': metadata['original_format'],
                'status': 'failed',
                'error': str(e)
            })

            raise Exception(f"Preprocessing failed for {filename}: {str(e)}")

    def _detect_encoding(self, file) -> str:
        """Detect file encoding"""
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

    def _convert_json(self, file, **kwargs) -> pd.DataFrame:
        """
        Convert JSON file to DataFrame.
        Handles multiple JSON structures: records, nested, lines.
        """
        try:
            file.seek(0)
            content = file.read()

            # Decode if bytes
            if isinstance(content, bytes):
                encoding = self._detect_encoding(file)
                file.seek(0)
                content = content.decode(encoding)

            # Try different JSON reading methods
            try:
                # Try as JSON lines (JSONL format)
                file.seek(0)
                df = pd.read_json(file, lines=True)
                return df
            except:
                pass

            try:
                # Try as standard JSON array/object
                data = json.loads(content)

                # Handle different structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Check if it's a nested structure with data key
                    if 'data' in data and isinstance(data['data'], list):
                        df = pd.DataFrame(data['data'])
                    elif 'records' in data and isinstance(data['records'], list):
                        df = pd.DataFrame(data['records'])
                    else:
                        # Try to normalize nested JSON
                        df = pd.json_normalize(data)
                        if len(df) == 0 and len(data) > 0:
                            # Convert dict to records format
                            df = pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON structure")

                return df

            except Exception as e:
                raise ValueError(f"JSON parsing failed: {str(e)}")

        except Exception as e:
            raise Exception(f"JSON conversion failed: {str(e)}")

    def _convert_parquet(self, file) -> pd.DataFrame:
        """Convert Parquet file to DataFrame"""
        try:
            df = pd.read_parquet(file, engine='pyarrow')
            return df
        except ImportError:
            # Try with fastparquet if pyarrow not available
            try:
                df = pd.read_parquet(file, engine='fastparquet')
                return df
            except:
                raise ImportError("Please install pyarrow or fastparquet: pip install pyarrow")
        except Exception as e:
            raise Exception(f"Parquet conversion failed: {str(e)}")

    def _convert_tsv(self, file) -> pd.DataFrame:
        """Convert TSV file to DataFrame"""
        try:
            encoding = self._detect_encoding(file)
            file.seek(0)
            df = pd.read_csv(file, sep='\t', encoding=encoding, low_memory=False)
            return df
        except Exception as e:
            raise Exception(f"TSV conversion failed: {str(e)}")

    def _convert_txt(self, file, delimiter: str = None, **kwargs) -> pd.DataFrame:
        """
        Convert TXT file to DataFrame.
        Supports custom delimiters.
        """
        try:
            encoding = self._detect_encoding(file)
            file.seek(0)

            # Try to detect delimiter if not provided
            if delimiter is None:
                sample = file.read(1000)
                file.seek(0)

                if isinstance(sample, bytes):
                    sample = sample.decode(encoding)

                # Common delimiters
                delimiters = [',', '\t', ';', '|', ' ']
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                delimiter = max(delimiter_counts, key=delimiter_counts.get)

            df = pd.read_csv(file, sep=delimiter, encoding=encoding, low_memory=False)
            return df

        except Exception as e:
            raise Exception(f"TXT conversion failed: {str(e)}")

    def _convert_feather(self, file) -> pd.DataFrame:
        """Convert Feather file to DataFrame"""
        try:
            # Feather requires file path, so save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.feather') as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            df = pd.read_feather(tmp_path)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

            return df

        except Exception as e:
            raise Exception(f"Feather conversion failed: {str(e)}")

    def _convert_hdf5(self, file, key: str = None, **kwargs) -> pd.DataFrame:
        """
        Convert HDF5 file to DataFrame.
        Requires key parameter if multiple datasets exist.
        """
        try:
            # HDF5 requires file path, so save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # If key not provided, try to get first key
            if key is None:
                with pd.HDFStore(tmp_path, 'r') as store:
                    keys = store.keys()
                    if len(keys) == 0:
                        raise ValueError("HDF5 file contains no datasets")
                    key = keys[0]

            df = pd.read_hdf(tmp_path, key=key)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

            return df

        except Exception as e:
            raise Exception(f"HDF5 conversion failed: {str(e)}. For HDF5 files with multiple datasets, specify 'key' parameter.")

    def _convert_pickle(self, file) -> pd.DataFrame:
        """Convert Pickle file to DataFrame"""
        try:
            file.seek(0)
            df = pd.read_pickle(file)

            # Ensure it's a DataFrame
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Pickle file does not contain a DataFrame")

            return df

        except Exception as e:
            raise Exception(f"Pickle conversion failed: {str(e)}")

    def _convert_sqlite(self, file, table_name: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        """
        Convert SQLite database to DataFrame.
        Requires either table_name or custom SQL query.
        """
        try:
            # SQLite requires file path, so save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            conn = sqlite3.connect(tmp_path)

            try:
                if query:
                    # Use custom query
                    df = pd.read_sql_query(query, conn)
                elif table_name:
                    # Read specific table
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                else:
                    # Get first table
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()

                    if len(tables) == 0:
                        raise ValueError("Database contains no tables")

                    first_table = tables[0][0]
                    df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)

                conn.close()

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

                return df

            except Exception as e:
                conn.close()
                raise e

        except Exception as e:
            raise Exception(f"SQLite conversion failed: {str(e)}. Specify 'table_name' or 'query' parameter.")

    def _convert_html(self, file, table_index: int = 0, **kwargs) -> pd.DataFrame:
        """
        Convert HTML table to DataFrame.
        Uses table_index to select which table (default: first table).
        """
        try:
            file.seek(0)
            content = file.read()

            # Decode if bytes
            if isinstance(content, bytes):
                encoding = self._detect_encoding(file)
                file.seek(0)
                content = content.decode(encoding)

            # Read all tables
            tables = pd.read_html(io.StringIO(content))

            if len(tables) == 0:
                raise ValueError("No tables found in HTML file")

            if table_index >= len(tables):
                raise ValueError(f"Table index {table_index} out of range. File contains {len(tables)} table(s).")

            df = tables[table_index]
            return df

        except Exception as e:
            raise Exception(f"HTML conversion failed: {str(e)}")

    def _convert_xml(self, file, **kwargs) -> pd.DataFrame:
        """
        Convert XML file to DataFrame.
        Handles simple XML structures with repeating elements.
        """
        try:
            file.seek(0)
            content = file.read()

            # Decode if bytes
            if isinstance(content, bytes):
                encoding = self._detect_encoding(file)
                file.seek(0)
                content = content.decode(encoding)

            # Parse XML
            root = ET.fromstring(content)

            # Extract data
            data = []

            # Try to find repeating elements
            children = list(root)
            if len(children) == 0:
                raise ValueError("XML file has no child elements")

            # Assume first level children are records
            for child in children:
                record = {}
                for elem in child:
                    # Handle simple text elements
                    if elem.text:
                        record[elem.tag] = elem.text
                    # Include attributes if any
                    if elem.attrib:
                        for key, val in elem.attrib.items():
                            record[f"{elem.tag}_{key}"] = val

                if record:  # Only add non-empty records
                    data.append(record)

            if len(data) == 0:
                raise ValueError("No data extracted from XML")

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            raise Exception(f"XML conversion failed: {str(e)}. Note: Only simple XML structures are supported.")

    def get_conversion_log(self) -> list:
        """Get conversion history log"""
        return self.conversion_log.copy()

    def clear_conversion_log(self):
        """Clear conversion history"""
        self.conversion_log = []


def get_format_requirements(format_type: str) -> Dict[str, Any]:
    """
    Get requirements and parameters for specific format.

    Args:
        format_type: File format type

    Returns:
        Dict with format-specific requirements
    """
    requirements = {
        'json': {
            'description': 'JSON file (array of objects, nested structure, or JSON lines)',
            'optional_params': ['lines (bool): True for JSON lines format'],
            'examples': ['[{"col1": "val1"}, {"col1": "val2"}]', '{"data": [...]}']
        },
        'parquet': {
            'description': 'Apache Parquet columnar storage format',
            'required_packages': ['pyarrow or fastparquet'],
            'examples': ['data.parquet']
        },
        'tsv': {
            'description': 'Tab-separated values',
            'examples': ['col1\tcol2\tval1\tval2']
        },
        'txt': {
            'description': 'Text file with delimiters',
            'optional_params': ['delimiter (str): Custom delimiter character'],
            'examples': ['col1|col2|col3']
        },
        'feather': {
            'description': 'Feather binary columnar format',
            'required_packages': ['pyarrow'],
            'examples': ['data.feather']
        },
        'hdf5': {
            'description': 'Hierarchical Data Format version 5',
            'optional_params': ['key (str): Dataset key if multiple datasets exist'],
            'required_packages': ['tables'],
            'examples': ['data.h5', 'data.hdf5']
        },
        'pickle': {
            'description': 'Pickle serialized DataFrame',
            'warning': 'Only load pickle files from trusted sources',
            'examples': ['data.pkl']
        },
        'sqlite': {
            'description': 'SQLite database',
            'optional_params': [
                'table_name (str): Specific table to read',
                'query (str): Custom SQL query'
            ],
            'examples': ['SELECT * FROM table_name']
        },
        'html': {
            'description': 'HTML file containing tables',
            'optional_params': ['table_index (int): Index of table to extract (default: 0)'],
            'examples': ['<table><tr><td>data</td></tr></table>']
        },
        'xml': {
            'description': 'XML file with simple repeating structure',
            'warning': 'Only simple XML structures supported',
            'examples': ['<root><record><col1>val1</col1></record></root>']
        }
    }

    return requirements.get(format_type.lower(), {'description': 'Format information not available'})
