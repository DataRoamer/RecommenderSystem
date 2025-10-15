# 📊 EDA Tool - Quick Start Instructions

## 🚀 Getting Started

### Step 1: Navigate to the Tool Directory
Open Command Prompt or Terminal and navigate to the EDA tool folder:
```bash
cd C:\Astreon\eda_tool
```

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Use any of these methods:

**Method A: Python Module (Recommended)**
```bash
python -m streamlit run app.py
```

**Method B: Double-click**
- Double-click on `run_app.bat` file in the folder

**Method C: Python Launcher**
```bash
python run_app.py
```

## 🌐 Accessing the Tool

1. After running, Streamlit will start a web server
2. Your browser will automatically open to: `http://localhost:8501`
3. If it doesn't open automatically, copy the URL from the terminal

## 📁 Using the Tool

### Upload Data
1. Click "Browse files" in the upload section
2. Select a CSV or Excel file (.csv, .xlsx, .xls)
3. The tool will automatically analyze your data

### Navigate Sections
Use the sidebar to navigate between:
- **📁 Data Upload**: Upload and validate files
- **📊 Data Overview**: Basic statistics and preview
- **📋 Data Quality**: Comprehensive quality analysis

### Test with Sample Data
Try the included Titanic dataset:
- File location: `tests\test_data\titanic.csv`
- This will demonstrate all features

## 🔧 Troubleshooting

### "streamlit is not recognized"
Use: `python -m streamlit run app.py` instead

### Module Import Errors
Run: `pip install -r requirements.txt`

### Browser Doesn't Open
Manually navigate to: `http://localhost:8501`

### Permission Errors
Run Command Prompt as Administrator

## 📊 Understanding Results

### Quality Score (0-100)
- **90-100**: Excellent (🟢)
- **75-89**: Good (🟡)
- **60-74**: Fair (🟠)
- **Below 60**: Critical Issues (🔴)

### Analysis Tabs
- **Overview**: Key metrics and data types
- **Missing Data**: Completeness analysis
- **Duplicates**: Data redundancy check
- **Outliers**: Anomaly detection
- **Statistics**: Detailed column statistics

## 🛑 Stopping the Tool

Press `Ctrl+C` in the terminal to stop the application

---

**Need Help?** Check the detailed README.md file for more information.