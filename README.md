# 📊 Automated Data Quality & EDA Tool

A comprehensive desktop application for automated data quality assessment and exploratory data analysis, built with Streamlit. This tool helps data scientists quickly assess dataset readiness for machine learning projects.

## 🚀 Features

### ✅ Currently Implemented

- **📁 Data Upload & Ingestion**
  - Support for CSV and Excel files (.xlsx, .xls)
  - Automatic encoding detection (UTF-8, Latin-1, CP1252)
  - Smart separator detection for CSV files
  - Data validation and error handling

- **📊 Data Overview & Profiling**
  - Basic dataset statistics (shape, memory usage, data types)
  - Column-wise information (missing values, unique counts, data types)
  - Data preview with flexible viewing options

- **📋 Comprehensive Data Quality Analysis**
  - **Missing Values Analysis**: Detailed missing value patterns, visualizations, and critical column identification
  - **Duplicate Detection**: Full and subset-based duplicate identification with examples
  - **Outlier Detection**: Z-score and IQR methods with interactive visualizations
  - **Data Type Consistency**: Mixed type detection and conversion recommendations
  - **Constant Features**: Identification of constant and near-constant features
  - **Quality Scoring**: Overall quality score (0-100) with breakdown by category

- **📈 Advanced Visualizations**
  - Missing value heatmaps and bar charts
  - Data type distribution charts
  - Outlier box plots
  - Interactive charts with Plotly integration

- **🧭 Professional UI/UX**
  - Sidebar navigation between sections
  - Tabbed interface for organized analysis
  - Progress tracking and status indicators
  - Responsive design with professional styling

### 🚧 Planned Features (Based on Your Guide)

- **🔍 Exploratory Data Analysis (EDA)**
  - Distribution analysis and normality tests
  - Correlation analysis with heatmaps
  - Feature relationships with target variables
  - Statistical summaries and insights

- **🎯 Target Variable Analysis**
  - Classification: class balance, imbalance detection
  - Regression: distribution analysis, transformation suggestions
  - Severity classification and recommendations

- **🛠️ Feature Engineering Recommendations**
  - Automatic feature type classification
  - Encoding strategy recommendations (One-Hot, Label, Target encoding)
  - Scaling recommendations (Standard, Robust, MinMax)
  - Feature creation suggestions
  - Feature selection guidance

- **🚨 Data Leakage Detection**
  - Target leakage identification
  - Temporal leakage checks
  - Duplicate feature detection
  - Train-test leakage assessment

- **📈 Model Readiness Scoring**
  - Comprehensive readiness assessment
  - Priority action items
  - Before/after improvement tracking

- **📄 Professional Reporting**
  - PDF reports with visualizations
  - Excel exports with multiple sheets
  - Feature configuration exports (JSON/YAML)
  - Python preprocessing code generation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   # If you have the files, navigate to the project directory
   cd eda_tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The application will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## 📊 Quick Start Guide

### 1. Upload Your Data
- Click on the file uploader in the sidebar or main area
- Select a CSV or Excel file from your computer
- The tool will automatically detect encoding and format

### 2. Navigate Through Analysis
Use the sidebar navigation to explore different sections:
- **📁 Data Upload**: Upload and validate your dataset
- **📊 Data Overview**: Basic statistics and data profiling
- **📋 Data Quality**: Comprehensive quality assessment with scoring

### 3. Interpret Results
- **Quality Score**: Overall assessment from 0-100
  - 90-100: Excellent (🟢)
  - 75-89: Good (🟡)
  - 60-74: Fair (🟠)
  - Below 60: Critical Issues (🔴)

- **Missing Data**: Check for completeness issues
- **Duplicates**: Identify potential data redundancy
- **Outliers**: Find anomalous values that might need attention
- **Statistics**: Understand your data distributions

## 📁 Project Structure

```
eda_tool/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and validation
│   └── data_quality.py         # Quality analysis functions
├── utils/
│   ├── __init__.py
│   └── visualizations.py       # Plotting and chart functions
├── config/
│   └── settings.py             # Configuration parameters
└── tests/
    ├── __init__.py
    └── test_data/               # Sample datasets
        └── titanic.csv          # Titanic dataset for testing
```

## 🎯 Example Use Cases

### 1. Customer Churn Prediction
- Upload customer data CSV
- Check for missing contact information
- Identify data quality issues before modeling
- Get recommendations for feature encoding

### 2. Sales Forecasting
- Analyze time series sales data
- Detect outliers in sales figures
- Check for data consistency across time periods
- Assess readiness for forecasting models

### 3. Medical Research Data
- Validate patient data completeness
- Identify inconsistent medical measurements
- Check for duplicate patient records
- Ensure data privacy compliance

## 🔧 Configuration

### Analysis Thresholds
You can adjust analysis thresholds in `config/settings.py`:

```python
OUTLIER_ZSCORE_THRESHOLD = 3      # Z-score threshold for outliers
HIGH_CORRELATION_THRESHOLD = 0.8   # Correlation threshold
MISSING_VALUE_CRITICAL_THRESHOLD = 0.5  # Critical missing data %
```

### Performance Settings
```python
LARGE_DATASET_THRESHOLD = 100000   # Rows threshold for sampling
SAMPLING_SIZE = 50000             # Sample size for large datasets
MAX_PLOT_POINTS = 10000           # Maximum points in plots
```

## 📋 Quality Assessment Methodology

### Quality Score Calculation (0-100 points)

1. **Missing Data Score (30 points)**
   - 0% missing: 30 points
   - <5% missing: 25 points
   - <15% missing: 20 points
   - <30% missing: 15 points
   - ≥30% missing: 10 points

2. **Duplicate Score (25 points)**
   - 0% duplicates: 25 points
   - <1% duplicates: 20 points
   - <5% duplicates: 15 points
   - ≥5% duplicates: 10 points

3. **Consistency Score (25 points)**
   - No constant features: 25 points
   - <10% constant features: 20 points
   - <20% constant features: 15 points
   - ≥20% constant features: 10 points

4. **Outlier Score (20 points)**
   - <1% outliers: 20 points
   - <5% outliers: 15 points
   - <10% outliers: 10 points
   - ≥10% outliers: 5 points

## 🔍 Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Ensure file is in CSV or Excel format
   - Check file is not corrupted
   - Verify file size is reasonable (<500MB recommended)

2. **Encoding Issues**
   - Tool automatically detects encoding
   - If issues persist, save CSV as UTF-8

3. **Memory Issues**
   - For large datasets (>100K rows), tool will automatically sample
   - Consider reducing dataset size for analysis

4. **Visualization Errors**
   - Usually caused by insufficient data
   - Check that numeric columns exist for numeric analysis

### Performance Tips

- For datasets >100K rows, expect longer analysis times
- Close other browser tabs to free up memory
- Use the sampling feature for initial exploration of large datasets

## 🚀 Future Development Roadmap

This tool is implementing a comprehensive 13-phase development plan:

**Phase 1-3**: ✅ **COMPLETED** - Basic functionality, data quality analysis
**Phase 4-6**: 🚧 **IN PROGRESS** - EDA, feature engineering, advanced analysis
**Phase 7-9**: 📋 **PLANNED** - Reporting, testing, documentation
**Phase 10-13**: 🔮 **FUTURE** - Deployment, distribution, maintenance

## 📞 Support & Feedback

For issues, suggestions, or contributions:
1. Create detailed issue descriptions
2. Include sample data (anonymized) when possible
3. Specify your operating system and Python version

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [Pandas](https://pandas.pydata.org/) for data manipulation
- Visualizations powered by [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and [Plotly](https://plotly.com/)
- Statistical analysis with [SciPy](https://scipy.org/) and [Scikit-learn](https://scikit-learn.org/)

---

**Ready to assess your data quality? Upload your dataset and get started! 📊✨**