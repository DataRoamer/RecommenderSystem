# Data Preprocessing Feature Guide

## Overview

The EDA Tool now includes comprehensive data preprocessing capabilities that automatically convert various file formats into pandas DataFrames for seamless analysis. This feature eliminates the need for manual data conversion and allows you to work with a wide variety of data sources.

---

## Supported File Formats

### Standard Formats (Direct Support)

#### 1. CSV (Comma-Separated Values)
- **Extensions:** `.csv`
- **Features:**
  - Automatic encoding detection
  - Multiple separator support (`,`, `;`, `\t`, `|`)
  - Handles large files efficiently
- **Example:**
  ```csv
  name,age,city
  John,25,New York
  Jane,30,San Francisco
  ```

#### 2. Excel
- **Extensions:** `.xlsx`, `.xls`
- **Features:**
  - Support for both modern and legacy Excel formats
  - Automatic sheet reading (first sheet by default)
- **Example:** Standard Excel workbooks

---

### Pre-processed Formats (Auto-converted)

#### 3. JSON (JavaScript Object Notation)
- **Extension:** `.json`
- **Supported Structures:**
  - **Array of objects:**
    ```json
    [
      {"name": "John", "age": 25},
      {"name": "Jane", "age": 30}
    ]
    ```
  - **Nested structure with data key:**
    ```json
    {
      "data": [
        {"name": "John", "age": 25},
        {"name": "Jane", "age": 30}
      ]
    }
    ```
  - **JSON Lines (JSONL):** One JSON object per line
    ```json
    {"name": "John", "age": 25}
    {"name": "Jane", "age": 30}
    ```
- **Features:**
  - Automatic structure detection
  - Support for nested JSON normalization
  - Encoding detection

#### 4. Parquet
- **Extension:** `.parquet`
- **Description:** Apache Parquet columnar storage format
- **Features:**
  - High compression and performance
  - Preserves data types
  - Supports both pyarrow and fastparquet engines
- **Use Case:** Large datasets with complex schemas

#### 5. TSV (Tab-Separated Values)
- **Extension:** `.tsv`
- **Description:** Similar to CSV but uses tabs as delimiters
- **Features:**
  - Automatic encoding detection
  - Handles quoted fields
- **Example:**
  ```tsv
  name\tage\tcity
  John\t25\tNew York
  Jane\t30\tSan Francisco
  ```

#### 6. TXT (Text Files)
- **Extension:** `.txt`
- **Features:**
  - Automatic delimiter detection (`,`, `\t`, `;`, `|`, ` `)
  - Custom delimiter support
  - Encoding detection
- **Use Case:** Text files with any consistent delimiter

#### 7. Feather
- **Extension:** `.feather`
- **Description:** Fast, lightweight binary columnar format
- **Features:**
  - Very fast read/write operations
  - Preserves pandas data types
  - Requires pyarrow
- **Use Case:** Fast data exchange between Python/R

#### 8. HDF5 (Hierarchical Data Format)
- **Extensions:** `.h5`, `.hdf5`
- **Features:**
  - Support for hierarchical data structures
  - Automatic key detection (first dataset if not specified)
  - Can store multiple datasets
- **Advanced Usage:** For files with multiple datasets, the first dataset is automatically selected
- **Use Case:** Scientific data, large numerical datasets

#### 9. Pickle
- **Extensions:** `.pkl`, `.pickle`
- **Description:** Python serialized DataFrame
- **Features:**
  - Preserves exact pandas DataFrame state
  - Includes index, data types, and metadata
- **⚠️ Warning:** Only load pickle files from trusted sources (security risk)
- **Use Case:** Saving/loading preprocessed pandas DataFrames

#### 10. SQLite Database
- **Extensions:** `.db`, `.sqlite`, `.sqlite3`
- **Features:**
  - Automatic table detection (first table by default)
  - Support for custom SQL queries
  - Reads all columns from selected table
- **Advanced Usage:**
  - To specify table: Add parameter `table_name="your_table"`
  - To use custom query: Add parameter `query="SELECT * FROM table WHERE condition"`
- **Use Case:** SQLite database files, local app databases

#### 11. HTML Tables
- **Extensions:** `.html`, `.htm`
- **Features:**
  - Extracts tables from HTML files
  - Automatic table detection (first table by default)
  - Handles complex table structures
- **Advanced Usage:** For files with multiple tables, the first table is automatically extracted
- **Use Case:** Web scraping results, exported reports

#### 12. XML
- **Extension:** `.xml`
- **Supported Structure:** Simple repeating elements
- **Example:**
  ```xml
  <root>
    <record>
      <name>John</name>
      <age>25</age>
      <city>New York</city>
    </record>
    <record>
      <name>Jane</name>
      <age>30</age>
      <city>San Francisco</city>
    </record>
  </root>
  ```
- **Features:**
  - Extracts repeating child elements
  - Handles element attributes
- **⚠️ Note:** Only simple XML structures are supported
- **Use Case:** Configuration files, simple data exports

---

## How It Works

### Automatic Processing Pipeline

1. **Upload File**
   - Select any supported file format
   - File is uploaded to the application

2. **Format Detection**
   - Automatic format detection based on file extension
   - Content analysis for ambiguous files (e.g., .txt)

3. **Pre-processing**
   - File is converted to pandas DataFrame
   - Data validation and cleaning
   - Metadata extraction

4. **Quality Checks**
   - Column validation
   - Missing value detection
   - Data type analysis

5. **Ready for Analysis**
   - DataFrame is ready for EDA
   - All standard features available

### User Interface

#### File Upload Section
```
📚 Supported File Formats & Pre-processing
[Expandable section with format information]

[Choose a data file]
[Drag and drop or browse]
```

#### Success Message (for preprocessed files)
```
✅ Successfully loaded your_file.json

🔄 Pre-processing Applied
- Original Format: JSON
- Converted to DataFrame: 1,000 rows × 25 columns
- Status: Success
```

---

## Installation

### Install Required Dependencies

```bash
cd C:\Astreon\eda_tool
pip install -r requirements.txt
```

### New Dependencies Added

```
pyarrow>=14.0.0       # For Parquet and Feather formats
tables>=3.9.0         # For HDF5 format
lxml>=5.0.0          # For HTML and XML parsing
html5lib>=1.1        # For better HTML table parsing
```

---

## Usage Examples

### Example 1: Loading a JSON File

**File: sales_data.json**
```json
[
  {"date": "2025-01-01", "product": "Widget", "sales": 100, "revenue": 1500.00},
  {"date": "2025-01-02", "product": "Gadget", "sales": 150, "revenue": 3000.00}
]
```

**Steps:**
1. Upload `sales_data.json`
2. Preprocessing automatically converts to DataFrame
3. All EDA features are now available

### Example 2: Loading a Parquet File

**File: large_dataset.parquet**

**Steps:**
1. Upload `large_dataset.parquet`
2. Automatic conversion with preserved data types
3. Efficient memory usage for large datasets

### Example 3: Loading an SQLite Database

**File: app_database.db**

**Steps:**
1. Upload `app_database.db`
2. First table is automatically selected
3. Data is ready for analysis

**Advanced:** To specify a table or query, you would need to use the data_loader module directly with parameters.

### Example 4: Loading HTML Tables

**File: report.html**
```html
<html>
  <table>
    <tr><th>Name</th><th>Value</th></tr>
    <tr><td>Item 1</td><td>100</td></tr>
    <tr><td>Item 2</td><td>200</td></tr>
  </table>
</html>
```

**Steps:**
1. Upload `report.html`
2. First table is automatically extracted
3. DataFrame is created from table data

---

## Technical Details

### DataPreprocessor Class

Located in: `modules/data_preprocessor.py`

**Key Methods:**
- `detect_format(file, filename)` - Auto-detect file format
- `is_supported_format(format_type)` - Check format support
- `preprocess_file(file, filename, **kwargs)` - Main conversion pipeline
- `get_conversion_log()` - Retrieve conversion history

**Individual Converters:**
- `_convert_json()` - JSON to DataFrame
- `_convert_parquet()` - Parquet to DataFrame
- `_convert_tsv()` - TSV to DataFrame
- `_convert_txt()` - TXT to DataFrame (with delimiter detection)
- `_convert_feather()` - Feather to DataFrame
- `_convert_hdf5()` - HDF5 to DataFrame
- `_convert_pickle()` - Pickle to DataFrame
- `_convert_sqlite()` - SQLite to DataFrame
- `_convert_html()` - HTML to DataFrame
- `_convert_xml()` - XML to DataFrame

### Integration Points

1. **data_loader.py**
   - Updated `load_data()` function
   - Integrated DataPreprocessor
   - Routing logic for different formats

2. **app.py**
   - Updated file uploader to accept new formats
   - Enhanced success messages with preprocessing info
   - Added format information expander

---

## Error Handling

### Common Issues and Solutions

#### Issue 1: "Unsupported file format"
- **Cause:** File extension not recognized
- **Solution:** Check that file extension matches supported formats

#### Issue 2: "Conversion resulted in empty DataFrame"
- **Cause:** File structure not compatible
- **Solution:** Verify file content and structure

#### Issue 3: JSON parsing failed
- **Cause:** Invalid JSON syntax
- **Solution:** Validate JSON file structure

#### Issue 4: HDF5 conversion failed
- **Cause:** Missing key or corrupted file
- **Solution:** Check HDF5 file integrity

#### Issue 5: SQLite - no tables found
- **Cause:** Empty database
- **Solution:** Verify database contains tables

#### Issue 6: XML - no data extracted
- **Cause:** Complex XML structure
- **Solution:** Only simple repeating structures are supported; consider converting to CSV

---

## Best Practices

### 1. File Size Considerations
- **Large files (>500MB):** Consider using Parquet or HDF5 for better performance
- **Memory usage:** Tool will warn if file exceeds 500MB

### 2. Format Selection
- **For speed:** Use Feather or Parquet
- **For compatibility:** Use CSV or Excel
- **For hierarchical data:** Use JSON or HDF5
- **For database exports:** Use SQLite or CSV

### 3. Data Quality
- **Always review preprocessing warnings**
- **Check for columns with all null values**
- **Verify data types after conversion**

### 4. Security
- **Never load pickle files from untrusted sources**
- **Review PII detection results**
- **Ensure proper authorization before processing sensitive data**

---

## Limitations

### Current Limitations

1. **SQLite:**
   - Automatic selection of first table only
   - Advanced queries not supported via UI
   - Multiple table joins require pre-processing

2. **HDF5:**
   - Automatic selection of first key only
   - Complex hierarchies may need manual extraction

3. **HTML:**
   - Only the first table is extracted
   - Complex nested tables may not parse correctly

4. **XML:**
   - Only simple repeating structures supported
   - Nested or complex XML requires pre-processing

5. **File Size:**
   - Very large files (>1GB) may cause memory issues
   - Consider chunking or sampling for analysis

### Workarounds

For complex scenarios:
1. **Pre-process externally:** Convert to CSV/Excel first
2. **Use Python directly:** Access data_preprocessor module with custom parameters
3. **Split large files:** Process in smaller chunks

---

## Future Enhancements

Planned improvements:
- [ ] Support for multiple SQL table selection via UI
- [ ] Advanced HDF5 key selection
- [ ] Multiple HTML table extraction
- [ ] Support for more complex XML structures
- [ ] ORC (Optimized Row Columnar) format support
- [ ] Avro format support
- [ ] Support for compressed files (zip, gzip, bz2)
- [ ] Data preview before full conversion
- [ ] Custom delimiter specification via UI
- [ ] Batch file processing

---

## Testing

### Test Files Preparation

Create test files for each format:

```python
import pandas as pd

# Sample data
df = pd.DataFrame({
    'name': ['John', 'Jane', 'Bob'],
    'age': [25, 30, 35],
    'city': ['New York', 'San Francisco', 'Chicago']
})

# Save in different formats
df.to_csv('test.csv', index=False)
df.to_excel('test.xlsx', index=False)
df.to_json('test.json', orient='records')
df.to_parquet('test.parquet')
df.to_feather('test.feather')
df.to_hdf('test.h5', key='data')
df.to_pickle('test.pkl')
df.to_html('test.html', index=False)

# For SQLite
import sqlite3
conn = sqlite3.connect('test.db')
df.to_sql('test_table', conn, index=False)
conn.close()
```

---

## Support

### Getting Help

For issues or questions:
- **Email:** contact@astreon.com.au
- **GitHub:** https://github.com/astreon-com-au/EDA_Tool

### Reporting Issues

When reporting preprocessing issues, include:
1. File format and extension
2. File size
3. Error message
4. Sample data structure (if not sensitive)

---

## Changelog

### Version 1.0 (Current)
- ✅ Initial implementation
- ✅ Support for 12 file formats
- ✅ Automatic format detection
- ✅ Integration with existing EDA features
- ✅ Error handling and validation
- ✅ User interface enhancements

---

## API Reference

### get_format_requirements(format_type: str)

Get detailed requirements for a specific format.

**Parameters:**
- `format_type` (str): File format identifier (e.g., 'json', 'parquet')

**Returns:**
- `dict`: Format requirements including description, optional parameters, and examples

**Example:**
```python
from modules.data_preprocessor import get_format_requirements

info = get_format_requirements('json')
print(info['description'])
# Output: JSON file (array of objects, nested structure, or JSON lines)
```

---

**Last Updated:** November 10, 2025
**Version:** 1.0
**Maintained By:** Astreon Development Team

---

*Generated as part of the EDA Tool Enhancement Project*
