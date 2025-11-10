"""
Test the data preprocessing module with generated test files
"""

import os
import sys
from modules.data_preprocessor import DataPreprocessor

def test_format(preprocessor, filename, format_type):
    """Test preprocessing for a specific format"""
    filepath = os.path.join('test_files', filename)

    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"Format: {format_type}")
    print('-' * 60)

    try:
        with open(filepath, 'rb') as f:
            df, metadata = preprocessor.preprocess_file(f, filename)

        print(f"[SUCCESS] Conversion successful!")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Column names: {', '.join(df.columns.tolist())}")
        print(f"  - Conversion status: {metadata.get('conversion_status', 'Unknown')}")

        if metadata.get('warnings'):
            print(f"  - Warnings: {len(metadata['warnings'])}")
            for warning in metadata['warnings']:
                print(f"    * {warning}")

        return True

    except Exception as e:
        print(f"[FAILED] Error: {str(e)}")
        return False

def main():
    print("="*60)
    print("Data Preprocessing Module Test")
    print("="*60)

    preprocessor = DataPreprocessor()

    # Test files to check
    test_files = [
        ('test.json', 'json'),
        ('test_lines.json', 'json'),
        ('test.parquet', 'parquet'),
        ('test.tsv', 'tsv'),
        ('test.txt', 'txt'),
        ('test.feather', 'feather'),
        ('test.h5', 'hdf5'),
        ('test.pkl', 'pickle'),
        ('test.db', 'sqlite'),
        ('test.html', 'html'),
        ('test.xml', 'xml'),
    ]

    results = {}

    for filename, format_type in test_files:
        success = test_format(preprocessor, filename, format_type)
        results[filename] = success

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed tests:")
        for filename, success in results.items():
            if not success:
                print(f"  - {filename}")

    if failed == 0:
        print("[SUCCESS] All preprocessing tests passed!")
    else:
        print(f"[WARNING] {failed} test(s) failed")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
