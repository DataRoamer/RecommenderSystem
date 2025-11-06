# AI Features Test Plan - EDA Tool v1.1.0
# Test Plan for EDA_tool_AI Branch

**Version:** 1.0
**Date:** October 31, 2024
**Branch:** EDA_tool_AI
**Tester:** QA Team

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Test Cases](#test-cases)
4. [Expected Results](#expected-results)
5. [Bug Report Template](#bug-report-template)
6. [Known Limitations](#known-limitations)

---

## Prerequisites

### System Requirements

- **Operating System:** Windows 10/11, macOS 10.15+, or Linux
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** 10 GB free (for models)
- **Python:** 3.8 or higher
- **Internet:** Required for initial Ollama and model download

### Software Requirements

- ✅ Git installed
- ✅ Python 3.8+ installed
- ✅ pip package manager
- ✅ Web browser (Chrome, Firefox, Edge, or Safari)

---

## Setup Instructions

### Step 1: Switch to AI Branch

```bash
# Navigate to eda_tool directory
cd eda_tool

# Switch to AI branch
git checkout EDA_tool_AI

# Verify you're on correct branch
git branch --show-current
# Expected output: EDA_tool_AI
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installations
pip list | grep -E "streamlit|requests|psutil"
```

### Step 3: Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# Expected: Browser opens to localhost:8501
```

---

## Test Cases

### 🧪 **Test Suite 1: Initial Application Launch**

#### TC-001: Verify App Starts Successfully

**Objective:** Ensure the application launches without errors on the AI branch

**Steps:**
1. Run `streamlit run app.py`
2. Wait for browser to open
3. Check console for errors

**Expected Result:**
- ✅ Browser opens to localhost:8501
- ✅ Welcome screen displays "AI-Powered EDA Tool"
- ✅ No errors in terminal
- ✅ Sidebar shows navigation menu
- ✅ AI Setup option visible in sidebar

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-002: Verify AI Features Listed in Welcome Screen

**Objective:** Confirm AI features are advertised on welcome screen

**Steps:**
1. View the welcome screen
2. Look for "AI Features (Privacy-First)" section

**Expected Result:**
- ✅ Section titled "🤖 AI Features (Privacy-First):" exists
- ✅ Lists: 100% Local Processing
- ✅ Lists: AI Chat Assistant
- ✅ Lists: Auto-Generated Insights
- ✅ Lists: Code Generation
- ✅ Lists: Smart Recommendations

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-003: Verify AI Status Badge in Sidebar

**Objective:** Check AI status badge display

**Steps:**
1. Look at the top of the sidebar
2. Check for AI status badge

**Expected Result:**
- ✅ Badge shows "🤖 AI: Not Available" OR "🤖 AI: Service offline"
- ✅ Badge is visible and styled properly
- ✅ No errors in console

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 2: AI Setup Wizard Access**

#### TC-004: Navigate to AI Setup

**Objective:** Access the AI Setup wizard

**Steps:**
1. In sidebar, click "🤖 AI Setup" button
2. Wait for page to load

**Expected Result:**
- ✅ Page changes to AI Setup wizard
- ✅ Title shows "🤖 AI Features Setup"
- ✅ Shows "Step 1: Ollama Installation"
- ✅ No errors displayed

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 3: Ollama Installation Detection**

#### TC-005: Test WITHOUT Ollama Installed

**Objective:** Verify app detects when Ollama is NOT installed

**Prerequisites:** Ollama must NOT be installed on test machine

**Steps:**
1. Navigate to AI Setup page
2. Observe Step 1 status

**Expected Result:**
- ✅ Shows "❌ Ollama is not installed on your system"
- ✅ Displays installation instructions
- ✅ Shows platform-specific instructions (Windows/Mac/Linux)
- ✅ No further steps visible
- ✅ Warning message: "After installing Ollama, please restart this application"

**Actual Result:**
[x] Pass
[ ] Fail
[ ] N/A (Ollama already installed)

**Notes:**
```
[Your observations here]
```

---

#### TC-006: Install Ollama

**Objective:** Install Ollama following provided instructions

**Steps:**
1. Follow installation instructions displayed in Step 1
2. For **Windows**: Download from https://ollama.ai/download/windows or use `winget install Ollama.Ollama`
3. For **Mac**: Download from https://ollama.ai/download/mac or use `brew install ollama`
4. For **Linux**: Run `curl -fsSL https://ollama.ai/install.sh | sh`
5. Verify installation: Run `ollama --version` in terminal

**Expected Result:**
- ✅ Ollama installs successfully
- ✅ `ollama --version` returns version number
- ✅ No errors during installation

**Actual Result:**
[x] Pass
[ ] Fail

**Ollama Version Installed:**
```
[Paste version output here]
```

**Notes:**
```
[Your observations here]
```

---

#### TC-007: Test WITH Ollama Installed

**Objective:** Verify app detects Ollama after installation

**Prerequisites:** Ollama must be installed (TC-006 completed)

**Steps:**
1. **Restart the Streamlit application** (IMPORTANT!)
   - Stop app: Press `Ctrl+C` in terminal
   - Restart: `streamlit run app.py`
2. Navigate to AI Setup page
3. Observe Step 1 status

**Expected Result:**
- ✅ Shows "✅ Ollama is installed! Version: [version]"
- ✅ Shows "✅ Ollama is running. Available models: [list]" OR "No models installed"
- ✅ Step 2 (System Information) is now visible
- ✅ Step 3 (Model Selection) is now visible

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 4: System Information Display**

#### TC-008: Verify System Information

**Objective:** Check system info displays correctly

**Steps:**
1. Navigate to AI Setup page (with Ollama installed)
2. Look at Step 2: System Information

**Expected Result:**
- ✅ Shows Operating System
- ✅ Shows Available RAM (in GB)
- ✅ Shows CPU Cores
- ✅ Shows RAM Status indicator:
  - 🟢 Excellent (16+ GB)
  - 🟡 Good (8-16 GB)
  - 🟠 Limited (4-8 GB)
  - 🔴 Low (<4 GB)

**Actual Result:**
[x] Pass
[ ] Fail

**System Info Detected:**
```
OS:
RAM:
CPU Cores:
RAM Status:
```

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 5: Model Management**

#### TC-009: View Available Models

**Objective:** Verify model cards display correctly

**Steps:**
1. Navigate to AI Setup page
2. Scroll to Step 3: Choose Your AI Model
3. Observe model cards

**Expected Result:**
- ✅ At least 4 model options visible:
  1. ⭐ Llama 3.1 8B (Recommended)
  2. Mistral 7B
  3. Phi-3 Mini
  4. Qwen 2.5 7B
- ✅ Each card shows:
  - Model name
  - Description
  - Size (e.g., "4.7 GB")
  - RAM Required
  - Quality stars (⭐)
  - Speed stars (⚡)
  - Install/Remove button

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-010: Download a Model (Llama 3.1 8B)

**Objective:** Download and install a model

**Prerequisites:** Ollama installed, no models installed yet

**Steps:**
1. Expand "⭐ Llama 3.1 8B" model card
2. Click "⬇️ Download 4.7 GB" button
3. Observe download progress
4. Wait for completion (may take 5-15 minutes depending on internet speed)

**Expected Result:**
- ✅ Shows "Starting download..." message
- ✅ Progress updates appear (may be console output)
- ✅ On completion: Shows "✅ Successfully downloaded llama3.1:8b"
- ✅ Balloons animation appears 🎈
- ✅ Page refreshes automatically
- ✅ Model card now shows "✅ Installed"
- ✅ "🗑️ Remove" button appears
- ✅ "🎯 Set as Default" button appears

**Actual Result:**
[ ] Pass
[ ] Fail

**Download Time:**
```
Started: [time]
Completed: [time]
Duration: [minutes]
```

**Notes:**
```
[Your observations here]
```

---

#### TC-011: Set Default Model

**Objective:** Set downloaded model as default

**Steps:**
1. After model is installed (TC-010)
2. Click "🎯 Set as Default" button
3. Observe result

**Expected Result:**
- ✅ Shows "✅ Llama 3.1 8B set as default" success message
- ✅ Page refreshes
- ✅ "Default Model" section appears below model cards
- ✅ Shows "Current default: llama3.1:8b"

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-012: Verify Model in Settings

**Objective:** Check model appears in sidebar settings

**Steps:**
1. Go to sidebar
2. Expand "⚙️ Settings" section
3. Scroll to "AI Settings"

**Expected Result:**
- ✅ "AI Settings" section visible
- ✅ Shows "Active Model" dropdown
- ✅ Dropdown contains "llama3.1:8b"
- ✅ Shows "Temperature" slider (0.0 to 1.0)
- ✅ Shows "Models Installed: 1"
- ✅ Shows "Queries Run: 0"
- ✅ Shows "⚙️ Manage Models" button

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-013: Verify AI Status Badge Updates

**Objective:** Confirm status badge shows model is ready

**Steps:**
1. After setting default model (TC-011)
2. Look at top of sidebar

**Expected Result:**
- ✅ Badge now shows "🤖 AI: Llama 3.1" (or similar)
- ✅ Badge color is green/success
- ✅ No longer shows "Not Available"

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-014: Download Additional Model (Optional)

**Objective:** Test downloading a second model

**Steps:**
1. Navigate to AI Setup
2. Expand "Phi-3 Mini" model card
3. Click "⬇️ Download 2.3 GB"
4. Wait for completion

**Expected Result:**
- ✅ Download proceeds successfully
- ✅ Both models now shown as installed
- ✅ Can switch between models in settings

**Actual Result:**
[x] Pass
[ ] Fail
[ ] Skipped

**Notes:**
```
[Your observations here]
```

---

#### TC-015: Remove a Model

**Objective:** Test model removal functionality

**Steps:**
1. Navigate to AI Setup
2. Find an installed model
3. Click "🗑️ Remove" button
4. Confirm removal

**Expected Result:**
- ✅ Shows removing progress
- ✅ Success message appears
- ✅ Page refreshes
- ✅ Model card now shows "⬇️ Download" instead of "✅ Installed"
- ✅ Model no longer appears in settings dropdown

**Actual Result:**
[x] Pass
[ ] Fail
[ ] Skipped

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 6: Error Handling**

#### TC-016: Test App Without Internet (After Setup)

**Objective:** Verify app works offline after setup

**Prerequisites:** At least one model installed

**Steps:**
1. Disconnect from internet
2. Refresh the app
3. Navigate through sections

**Expected Result:**
- ✅ App loads successfully
- ✅ AI status badge shows model is available
- ✅ No errors about internet connection
- ✅ AI Setup page accessible
- ✅ Model information displays correctly

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-017: Test with Ollama Service Stopped

**Objective:** Verify graceful degradation when Ollama stops

**Prerequisites:** Ollama installed

**Steps:**
1. Stop Ollama service:
   - **Windows**: Stop in Task Manager
   - **Mac/Linux**: `killall ollama`
2. Refresh the app
3. Check AI status badge
4. Navigate to AI Setup

**Expected Result:**
- ✅ AI status badge shows "🤖 AI: Service offline"
- ✅ AI Setup shows warning: "⚠️ Ollama service is not running"
- ✅ Shows instruction: "Please start Ollama service and refresh"
- ✅ No Python errors or crashes
- ✅ Rest of app still works (data upload, analysis, etc.)

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 7: Navigation and UI**

#### TC-018: Test All Navigation Buttons

**Objective:** Verify all navigation works

**Steps:**
1. Click through each section in sidebar:
   - 📁 Data Upload
   - 🤖 AI Setup
   - 📊 Data Overview
   - 📋 Data Quality
   - 🔍 EDA
   - 🎯 Target Analysis
   - 🛠️ Feature Engineering
   - 🚨 Leakage Detection
   - 📈 Model Readiness
   - 📄 Reports

**Expected Result:**
- ✅ Each section loads without errors
- ✅ Page transitions smoothly
- ✅ No broken navigation
- ✅ Can return to AI Setup from any section

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

#### TC-019: Test Settings Expander

**Objective:** Verify settings section works

**Steps:**
1. In sidebar, click "⚙️ Settings"
2. Try adjusting sliders
3. Check AI Settings subsection

**Expected Result:**
- ✅ Expander opens/closes smoothly
- ✅ All settings controls are functional
- ✅ AI Settings visible (if model installed)
- ✅ Temperature slider works (0.0 to 1.0)
- ✅ Model dropdown works
- ✅ "⚙️ Manage Models" button opens AI Setup

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 8: Data Upload with AI**

#### TC-020: Upload Dataset with AI Enabled

**Objective:** Ensure data upload still works with AI features

**Steps:**
1. Navigate to "📁 Data Upload"
2. Upload a test dataset (use `titanic.csv` if available)
3. Observe upload process

**Expected Result:**
- ✅ File uploads successfully
- ✅ Data quality report generates
- ✅ EDA report generates
- ✅ No AI-related errors
- ✅ Can navigate to other sections
- ✅ AI badge still shows in sidebar

**Actual Result:**
[x] Pass
[ ] Fail

**Notes:**
```
[Your observations here]
```

---

### 🧪 **Test Suite 9: Cross-Platform Testing**

#### TC-021: Windows-Specific Tests

**Platform:** Windows 10/11

**Steps:**
1. Run full test suite on Windows
2. Note any Windows-specific issues

**Expected Result:**
- ✅ All tests pass on Windows
- ✅ Ollama installation works
- ✅ Model download works
- ✅ No path or permission errors

**Actual Result:**
[x] Pass
[ ] Fail
[ ] N/A (Not on Windows)

**Windows Version:**
```
[Your Windows version]
```

**Notes:**
```
[Your observations here]
```

---

#### TC-022: macOS-Specific Tests

**Platform:** macOS 10.15+

**Steps:**
1. Run full test suite on macOS
2. Note any Mac-specific issues

**Expected Result:**
- ✅ All tests pass on macOS
- ✅ Ollama installation works
- ✅ Model download works
- ✅ No permission errors

**Actual Result:**
[ ] Pass
[ ] Fail
[ ] N/A (Not on Mac)

**macOS Version:**
```
[Your macOS version]
```

**Notes:**
```
[Your observations here]
```

---

#### TC-023: Linux-Specific Tests

**Platform:** Linux (Ubuntu, Debian, etc.)

**Steps:**
1. Run full test suite on Linux
2. Note any Linux-specific issues

**Expected Result:**
- ✅ All tests pass on Linux
- ✅ Ollama installation works
- ✅ Model download works
- ✅ No permission errors

**Actual Result:**
[ ] Pass
[ ] Fail
[ ] N/A (Not on Linux)

**Linux Distribution:**
```
[Your distro and version]
```

**Notes:**
```
[Your observations here]
```

---

## Expected Results Summary

### ✅ **Should Work Perfectly:**

1. **AI Setup Wizard**
   - Ollama detection
   - System info display
   - Model card display
   - Installation instructions

2. **Model Management**
   - Model download
   - Model removal
   - Default model selection
   - Settings integration

3. **UI Integration**
   - AI status badge
   - Navigation menu
   - Settings expander
   - Welcome screen updates

4. **Error Handling**
   - Graceful degradation if Ollama not installed
   - Works offline after setup
   - No crashes or Python errors

5. **Existing Features**
   - Data upload still works
   - All existing analysis features work
   - No regression bugs

---

## Known Limitations

### 🚧 **Expected Limitations (Not Bugs):**

1. **AI Chat Not Yet Implemented**
   - AI Chat Assistant button/page doesn't exist yet
   - This is Phase 1 remaining work

2. **Auto-Insights Not Yet Implemented**
   - No auto-generated insights on data upload
   - Coming in next development phase

3. **No Actual AI Queries Yet**
   - Models download but can't be used yet
   - Chat interface not built
   - This is expected - we're testing setup only

4. **Model Download Progress**
   - Progress may only show in terminal/console
   - UI progress bar is simplified
   - Still functional, just not as detailed

5. **First-Time Model Download**
   - Can take 5-20 minutes depending on internet
   - This is normal for 4-5 GB downloads

---

## Bug Report Template

If you find a bug, please report using this template:

```markdown
### Bug Report

**Bug ID:** BUG-[number]

**Test Case:** TC-[number]

**Severity:**
[ ] Critical (App crashes)
[ ] High (Feature doesn't work)
[ ] Medium (Feature works but has issues)
[ ] Low (Cosmetic/UI issue)

**Description:**
[Clear description of the issue]

**Steps to Reproduce:**
1.
2.
3.

**Expected Result:**
[What should happen]

**Actual Result:**
[What actually happened]

**Screenshots:**
[Attach if applicable]

**Environment:**
- OS: [Windows/Mac/Linux]
- OS Version: [version]
- Python Version: [version]
- Ollama Version: [version if applicable]
- RAM: [GB]

**Console Output/Error Messages:**
```
[Paste any error messages here]
```

**Additional Notes:**
[Any other relevant information]
```

---

## Test Completion Checklist

### Phase 1: Basic Setup (All Testers)
- [ ] TC-001: App launches
- [ ] TC-002: Welcome screen
- [ ] TC-003: AI status badge
- [ ] TC-004: Navigate to AI Setup
- [ ] TC-005: Ollama not installed detection
- [ ] TC-006: Install Ollama
- [ ] TC-007: Ollama installed detection
- [ ] TC-008: System information
- [ ] TC-009: View models
- [ ] TC-010: Download model
- [ ] TC-011: Set default model
- [ ] TC-012: Model in settings
- [ ] TC-013: Status badge updates

### Phase 2: Advanced (Optional)
- [ ] TC-014: Download second model
- [ ] TC-015: Remove model
- [ ] TC-016: Offline test
- [ ] TC-017: Ollama stopped test
- [ ] TC-018: All navigation
- [ ] TC-019: Settings expander
- [ ] TC-020: Data upload with AI

### Phase 3: Platform-Specific
- [ ] TC-021: Windows tests
- [ ] TC-022: macOS tests
- [ ] TC-023: Linux tests

---

## 📊 Test Results Summary

**Total Test Cases:** 23
**Passed:** ___
**Failed:** ___
**Skipped:** ___
**Pass Rate:** ___%

**Critical Bugs Found:** ___
**High Priority Bugs:** ___
**Medium Priority Bugs:** ___
**Low Priority Bugs:** ___

**Overall Assessment:**
```
[Your overall assessment of the AI features quality]
```

**Recommendation:**
[ ] Ready for production
[ ] Minor fixes needed
[ ] Major fixes needed
[ ] Not ready

---

## 📝 Tester Information

**Tester Name:** _______________
**Test Date:** _______________
**Test Duration:** _______________
**Test Environment:** _______________

**Signature:** _______________

---

## 🚀 Next Steps After Testing

Once testing is complete:

1. **Report Results:** Share this document with development team
2. **Log Bugs:** Create issues for any bugs found
3. **Prioritize Fixes:** Development team will prioritize bug fixes
4. **Retest:** After fixes, retest failed cases
5. **Approve Merge:** If all tests pass, approve merge to main branch

---

**End of Test Plan**

*Generated by: Claude Code*
*Branch: EDA_tool_AI*
*Version: 1.0*
