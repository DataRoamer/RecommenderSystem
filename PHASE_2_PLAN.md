# Phase 2: Advanced AI Features - Detailed Implementation Plan

**Project:** EDA Tool AI Integration
**Branch:** EDA_tool_AI
**Document Version:** 1.0
**Created:** November 7, 2024
**Status:** Planning Phase

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Phase 2a: Smart Data Cleaning Recommendations](#phase-2a-smart-data-cleaning-recommendations)
3. [Phase 2b: AI-Powered Feature Engineering](#phase-2b-ai-powered-feature-engineering)
4. [Phase 2c: Anomaly Explanation with Root Cause](#phase-2c-anomaly-explanation-with-root-cause)
5. [Phase 2d: Executive Report Generator](#phase-2d-executive-report-generator)
6. [Phase 2e: Code Explanation & Documentation](#phase-2e-code-explanation--documentation)
7. [Timeline & Resources](#timeline--resources)
8. [Success Metrics](#success-metrics)

---

## Overview

### Phase 2 Goals

Phase 2 builds upon the foundation established in Phase 1 (Chat Assistant, Insights, NL Query) by adding **advanced AI-powered features** that provide actionable recommendations and automate complex data science tasks.

### Key Principles

- ✅ **Privacy-First:** All processing remains local (no cloud APIs)
- ✅ **Explainable:** Every recommendation includes rationale
- ✅ **Safe:** Preview changes before applying, undo capability
- ✅ **User-Controlled:** Users approve all data modifications
- ✅ **Educational:** Teach users best practices through explanations

### Phase Priorities

1. **Phase 2a:** Smart Data Cleaning (HIGH) - Start here
2. **Phase 2b:** Feature Engineering (HIGH)
3. **Phase 2c:** Anomaly Explanation (MEDIUM)
4. **Phase 2d:** Report Generator (MEDIUM)
5. **Phase 2e:** Code Explanation (LOW)

---

## Phase 2a: Smart Data Cleaning Recommendations

### 🎯 Objective

Provide AI-powered, context-aware data cleaning recommendations with one-click fixes and comprehensive explanations.

### 📊 Priority: HIGH ⭐⭐⭐⭐

### ⏱️ Estimated Time: 20-25 hours

---

### Step-by-Step Implementation Plan

#### **Step 1: Create Core Module (3-4 hours)**

**File:** `modules/ai/data_cleaning_advisor.py`

**Tasks:**
- [ ] Create `DataCleaningAdvisor` class
- [ ] Implement `analyze_quality_issues()` method
- [ ] Implement `generate_recommendations()` method
- [ ] Create recommendation data structure
- [ ] Add confidence scoring system

**Code Structure:**
```python
class DataCleaningAdvisor:
    def __init__(self, model_name: str = 'phi3:mini'):
        pass

    def analyze_quality_issues(self, df, quality_report):
        # Detect issues from quality report
        pass

    def generate_recommendations(self, issues):
        # Use LLM to generate recommendations
        pass

    def apply_fix(self, df, fix_type, params):
        # Apply cleaning operation
        pass
```

**Deliverables:**
- Working `data_cleaning_advisor.py` module
- Unit tests for core functions
- Documentation

---

#### **Step 2: Issue Detection & Classification (4-5 hours)**

**Tasks:**
- [ ] **Invalid Values Detector**
  - Negative values in age/price columns
  - Out-of-range values (e.g., age > 150)
  - Invalid formats (emails, phone numbers)
  - Type mismatches

- [ ] **Duplicate Detector**
  - Exact duplicates (all columns match)
  - Fuzzy duplicates (similar but not exact)
  - Duplicate detection by subset of columns

- [ ] **Outlier Analyzer**
  - Statistical outliers (Z-score, IQR)
  - Context-aware analysis
  - Legitimate vs. erroneous outliers

- [ ] **Missing Value Strategist**
  - Pattern analysis (MCAR, MAR, MNAR)
  - Column-specific strategies
  - Correlation with other columns

- [ ] **Format Inconsistency Detector**
  - Date format variations
  - String case inconsistencies
  - Whitespace issues
  - Encoding problems

**Data Structure:**
```python
Issue = {
    'id': str,
    'type': 'invalid_values' | 'duplicates' | 'outliers' | 'missing' | 'format',
    'severity': 'critical' | 'high' | 'medium' | 'low',
    'column': str,
    'affected_rows': int,
    'description': str,
    'examples': List[Any]
}
```

---

#### **Step 3: Recommendation Engine (5-6 hours)**

**Tasks:**
- [ ] **Recommendation Generator**
  - Map issues to fix strategies
  - Generate multiple options per issue
  - Confidence scoring
  - Impact assessment

- [ ] **LLM Integration**
  - Create cleaning recommendation prompt
  - Format issue context for LLM
  - Parse LLM recommendations
  - Enhance with domain knowledge

- [ ] **Fix Strategies Library**
  - Replace with mean/median/mode
  - Remove duplicates
  - Cap/floor outliers
  - Standardize formats
  - Imputation strategies
  - Flag for manual review

**Recommendation Structure:**
```python
Recommendation = {
    'issue_id': str,
    'fix_type': str,
    'description': str,
    'rationale': str,  # AI-generated explanation
    'confidence': float,  # 0.0-1.0
    'impact': str,  # What changes
    'preview': Dict,  # Before/after examples
    'alternatives': List[Recommendation]
}
```

---

#### **Step 4: Preview & Apply System (3-4 hours)**

**Tasks:**
- [ ] **Preview Generator**
  - Show before/after comparison
  - Impact statistics (rows affected, values changed)
  - Visual diff for small datasets

- [ ] **Safe Apply System**
  - Create backup of original data
  - Apply changes to copy first
  - Validate results
  - Commit or rollback

- [ ] **Undo Stack**
  - Track all applied changes
  - Store undo information
  - Implement undo operation
  - Multiple undo levels

**Implementation:**
```python
def preview_fix(df, recommendation):
    # Show what will change
    return {
        'affected_rows': int,
        'before_sample': List,
        'after_sample': List,
        'statistics': Dict
    }

def apply_fix(df, recommendation, create_backup=True):
    # Apply with safety checks
    pass

def undo_last_fix():
    # Revert to previous state
    pass
```

---

#### **Step 5: UI Implementation (4-5 hours)**

**File:** `modules/ai/data_cleaning_advisor.py` (add `display_data_cleaning()` function)

**Tasks:**
- [ ] **Main Interface**
  - "Analyze Data Quality" button
  - Issue list with severity indicators
  - Expandable issue details

- [ ] **Issue Display**
  - Issue cards with:
    - Type badge (color-coded by severity)
    - Affected column & row count
    - Description
    - AI-generated explanation
    - Preview section
    - Action buttons

- [ ] **Action Buttons**
  - [Apply Fix] - Execute recommendation
  - [Preview] - Show detailed preview
  - [Ignore] - Skip this issue
  - [Custom...] - User specifies fix
  - [Explain More] - Get detailed AI explanation

- [ ] **Batch Operations**
  - Select multiple issues
  - "Apply All" with confirmation
  - Progress indicator

- [ ] **History Panel**
  - Show applied fixes
  - Undo button for each fix
  - Statistics (issues fixed, rows affected)

**UI Mockup:**
```
🧹 Smart Data Cleaning

[Analyze Data Quality]  [View History]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 CRITICAL: Invalid Values in 'age' Column
📍 Column: age | 🔢 Affected: 5 rows

Problem:
• 5 records have negative ages (values: -23, -15, -8, -45, -2)

Impact:
• Will cause errors in age-based analysis
• Statistical calculations will be incorrect

Recommendation:
• Replace with median age (38 years)

Confidence: ⭐⭐⭐⭐⭐ HIGH

[Apply Fix]  [Preview]  [Ignore]  [Custom...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟡 MEDIUM: Duplicate Records
📍 All columns | 🔢 Affected: 234 rows

Problem:
• 180 exact duplicates (same values in all columns)
• 54 fuzzy duplicates (similar names, same email)

Recommendation:
• Remove exact duplicates (keep first occurrence)
• Flag fuzzy duplicates for manual review

Confidence: ⭐⭐⭐⭐ HIGH for exact, ⭐⭐ MEDIUM for fuzzy

[Remove Exact]  [Review Fuzzy]  [Show Details]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

#### **Step 6: Integration (2 hours)**

**Tasks:**
- [ ] Update `modules/ai/__init__.py`
- [ ] Add navigation item in `app.py`
- [ ] Add routing for cleaning section
- [ ] Update welcome screen with new feature

**Navigation:**
- Position: After "🔍 EDA" (before AI Chat)
- Label: "🧹 Data Cleaning"

---

#### **Step 7: Testing (3-4 hours)**

**Test Cases:**
- [ ] Test with dataset containing invalid values
- [ ] Test with duplicates (exact and fuzzy)
- [ ] Test with outliers
- [ ] Test with missing values
- [ ] Test format inconsistencies
- [ ] Test preview system
- [ ] Test apply and undo operations
- [ ] Test batch operations
- [ ] Test with edge cases (empty dataset, no issues)
- [ ] Test memory efficiency with large datasets

**Datasets for Testing:**
- Titanic (missing values)
- Customer data (duplicates, format issues)
- Sales data (outliers)
- Synthetic data with known issues

---

### Deliverables

1. ✅ `modules/ai/data_cleaning_advisor.py` (300-400 lines)
2. ✅ UI function `display_data_cleaning()` (200-300 lines)
3. ✅ Navigation integration
4. ✅ Test suite
5. ✅ Documentation
6. ✅ Example screenshots

### Success Metrics

- Detects 90%+ of common data quality issues
- Recommendations are accurate and actionable
- Preview system clearly shows changes
- Undo works reliably
- User feedback is positive

---

## Phase 2b: AI-Powered Feature Engineering

### 🎯 Objective

Provide intelligent feature engineering recommendations based on target variable analysis and domain knowledge.

### 📊 Priority: HIGH ⭐⭐⭐⭐

### ⏱️ Estimated Time: 20-25 hours

---

### Step-by-Step Implementation Plan

#### **Step 1: Create Core Module (3-4 hours)**

**File:** `modules/ai/feature_engineering_advisor.py`

**Tasks:**
- [ ] Create `FeatureEngineeringAdvisor` class
- [ ] Implement target analysis
- [ ] Implement feature correlation analysis
- [ ] Create recommendation scoring system

---

#### **Step 2: Feature Analysis (5-6 hours)**

**Analysis Types:**

- [ ] **Binning Analysis**
  - Optimal bin count determination
  - Equal-width vs equal-frequency
  - Domain-specific bins (age groups, income brackets)

- [ ] **Transformation Analysis**
  - Log transformation for skewed data
  - Square root for count data
  - Box-Cox for normalization
  - Polynomial features for non-linear relationships

- [ ] **Interaction Features**
  - Multiplicative interactions (price × quantity)
  - Ratio features (debt/income)
  - Difference features (actual - predicted)

- [ ] **Time-Based Features**
  - Day of week, month, quarter
  - Weekend/weekday flags
  - Seasonal indicators
  - Time since reference date

- [ ] **Categorical Encoding**
  - One-hot encoding
  - Target encoding
  - Frequency encoding
  - Ordinal encoding

- [ ] **Domain-Specific Features**
  - Use LLM for domain knowledge
  - Industry-specific transformations
  - Business logic features

---

#### **Step 3: Recommendation Generator (4-5 hours)**

**Tasks:**
- [ ] Analyze feature importance potential
- [ ] Generate transformation suggestions
- [ ] Create new feature proposals
- [ ] Use LLM for rationale generation
- [ ] Code generation for transformations

**LLM Prompt:**
```
Analyze these features and target variable:
- Target: customer_churn (binary)
- Features: age, income, account_balance, num_transactions

Suggest feature engineering transformations that would
improve model performance. Include:
1. Transformation type
2. Rationale (why it helps)
3. Expected impact
4. Python/pandas code
```

---

#### **Step 4: Code Generation (3-4 hours)**

**Tasks:**
- [ ] Generate pandas code for each transformation
- [ ] Validate generated code
- [ ] Test on sample data
- [ ] Add error handling

**Example Output:**
```python
# Age Binning - Creates age groups for better pattern detection
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['Young', 'Adult', 'Middle-aged', 'Senior', 'Elderly']
)

# Income-to-Debt Ratio - Key financial health indicator
df['debt_to_income_ratio'] = df['total_debt'] / (df['annual_income'] + 1)

# Transaction Velocity - Activity level indicator
df['transactions_per_month'] = df['num_transactions'] / df['account_age_months']
```

---

#### **Step 5: UI Implementation (4-5 hours)**

**Interface Components:**
- [ ] Feature list with current statistics
- [ ] Recommendation cards
- [ ] Preview transformed features
- [ ] Side-by-side comparison (original vs engineered)
- [ ] Apply/ignore buttons
- [ ] Batch apply functionality

**UI Mockup:**
```
🛠️ AI Feature Engineering

Target Variable: customer_churn (binary)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⭐⭐⭐⭐⭐ HIGHLY RECOMMENDED

Age Binning
Current: Continuous (18-87)
Suggested: Categorical age groups

Rationale:
• Churn patterns differ significantly by age group
• Non-linear relationship with target
• Improves model interpretability

Expected Impact: +12% model accuracy

[Show Code]  [Preview]  [Apply]  [Skip]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

#### **Step 6: Integration & Testing (3-4 hours)**

**Tasks:**
- [ ] Integrate with existing feature engineering module
- [ ] Add navigation
- [ ] Test with various datasets and targets
- [ ] Validate generated features
- [ ] Performance testing

---

### Deliverables

1. ✅ `modules/ai/feature_engineering_advisor.py` (400-500 lines)
2. ✅ UI function `display_feature_engineering_ai()` (250-300 lines)
3. ✅ Integration with app.py
4. ✅ Test suite
5. ✅ Documentation

---

## Phase 2c: Anomaly Explanation with Root Cause

### 🎯 Objective

Provide AI-generated explanations for outliers and anomalies, helping users understand whether to keep, remove, or investigate.

### 📊 Priority: MEDIUM ⭐⭐⭐

### ⏱️ Estimated Time: 15-18 hours

---

### Step-by-Step Implementation Plan

#### **Step 1: Create Explainer Module (3-4 hours)**

**File:** `modules/ai/anomaly_explainer.py`

**Tasks:**
- [ ] Create `AnomalyExplainer` class
- [ ] Implement outlier context builder
- [ ] Create explanation generator
- [ ] Add action recommendation system

---

#### **Step 2: Context Analysis (4-5 hours)**

**Analysis Features:**

- [ ] **Pattern Detection**
  - Cluster analysis of outliers
  - Temporal patterns (e.g., weekend spikes)
  - Correlation with other variables

- [ ] **Statistical Context**
  - Z-score, IQR calculations
  - Distribution comparison
  - Percentile ranking

- [ ] **Domain Knowledge**
  - Use LLM for industry insights
  - Common causes in domain
  - Business context

---

#### **Step 3: Explanation Generation (3-4 hours)**

**LLM Prompt:**
```
Outlier detected:
- Column: salary
- Value: $250,000
- Z-score: 4.5
- Context: Tech industry, 15 years experience, Senior role
- Dataset: Employee records

Explain:
1. Is this likely legitimate or an error?
2. What could cause this value?
3. Recommendation: keep, remove, or investigate?
```

**Output Format:**
```
Analysis:
• This salary is 4.5 standard deviations above mean
• Value is within expected range for senior tech roles
• Correlates with years of experience and position level

Likely Cause:
• Legitimate high earner in senior position
• Tech industry typically pays above average

Recommendation: KEEP
• Value appears legitimate given context
• Removing would introduce bias
• Consider as separate segment in analysis
```

---

#### **Step 4: Interactive UI (4-5 hours)**

**Features:**
- [ ] Click on outlier visualization to get explanation
- [ ] Batch explanation for multiple outliers
- [ ] Action buttons (Keep/Remove/Flag)
- [ ] Explanation history
- [ ] Export explanations

**UI Mockup:**
```
🔍 Outlier Analysis: salary

Value: $250,000
Z-score: 4.5 (Extreme outlier)
Percentile: 99.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 AI Analysis:

Legitimacy: LIKELY LEGITIMATE ✅

Evidence:
• Correlates with senior position level (r=0.85)
• Within expected range for 15+ years experience
• Tech industry premium well-documented

Pattern Detection:
• 12 similar values cluster together
• All associated with senior roles
• No temporal anomalies detected

Recommendation: KEEP ⭐⭐⭐⭐
Rationale: Legitimate data point representing high earners.
Removing would introduce bias in salary analysis.

[Keep]  [Remove]  [Flag for Review]  [Explain More]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

#### **Step 5: Integration & Testing (2-3 hours)**

---

### Deliverables

1. ✅ `modules/ai/anomaly_explainer.py` (250-300 lines)
2. ✅ Interactive UI integration
3. ✅ Batch explanation feature
4. ✅ Documentation

---

## Phase 2d: Executive Report Generator

### 🎯 Objective

Generate professional, narrative reports summarizing all analyses in business-friendly language.

### 📊 Priority: MEDIUM ⭐⭐⭐

### ⏱️ Estimated Time: 12-15 hours

---

### Step-by-Step Implementation Plan

#### **Step 1: Report Builder Module (3-4 hours)**

**File:** `modules/ai/report_generator.py`

**Tasks:**
- [ ] Create `ReportGenerator` class
- [ ] Implement section generators
- [ ] Create report templates
- [ ] Add formatting system

---

#### **Step 2: Content Generation (4-5 hours)**

**Report Sections:**

- [ ] **Executive Summary** (AI-generated)
  - 3-5 key findings
  - Overall data quality score
  - Major recommendations
  - Business impact

- [ ] **Data Quality Assessment**
  - Missing data analysis
  - Outlier summary
  - Duplicate detection results
  - Format issues

- [ ] **Statistical Analysis**
  - Distribution summaries
  - Correlation insights
  - Trend analysis

- [ ] **Recommendations**
  - Priority actions
  - Implementation steps
  - Expected impact

- [ ] **Appendix**
  - Detailed statistics
  - Methodology
  - Definitions

---

#### **Step 3: Export Functionality (3-4 hours)**

**Export Formats:**
- [ ] PDF (using reportlab or weasyprint)
- [ ] HTML (styled with CSS)
- [ ] Markdown
- [ ] Word (using python-docx)

---

#### **Step 4: UI & Integration (2-3 hours)**

**Features:**
- [ ] Report preview
- [ ] Customize sections
- [ ] Download buttons
- [ ] Email report option (future)

---

### Deliverables

1. ✅ `modules/ai/report_generator.py` (300-400 lines)
2. ✅ Report templates
3. ✅ Export functionality
4. ✅ UI integration

---

## Phase 2e: Code Explanation & Documentation

### 🎯 Objective

Provide AI-powered code explanations for generated code, helping users learn and understand.

### 📊 Priority: LOW ⭐⭐

### ⏱️ Estimated Time: 10-12 hours

---

### Step-by-Step Implementation Plan

#### **Step 1: Code Explainer Module (3-4 hours)**

**File:** `modules/ai/code_explainer.py`

**Features:**
- [ ] Line-by-line explanations
- [ ] Complexity analysis
- [ ] Performance suggestions
- [ ] Alternative approaches

---

#### **Step 2: UI Integration (3-4 hours)**

**Features:**
- [ ] "Explain this code" button
- [ ] Inline annotations
- [ ] Hover tooltips
- [ ] Copy with comments

---

#### **Step 3: Documentation Generator (3-4 hours)**

**Features:**
- [ ] Function docstring generation
- [ ] Usage examples
- [ ] Parameter descriptions
- [ ] Return value documentation

---

### Deliverables

1. ✅ `modules/ai/code_explainer.py` (200-250 lines)
2. ✅ UI integration
3. ✅ Documentation

---

## Timeline & Resources

### Overall Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 2a | 20-25 hours | Week 1 | Week 1-2 |
| Phase 2b | 20-25 hours | Week 2 | Week 2-3 |
| Phase 2c | 15-18 hours | Week 3 | Week 3-4 |
| Phase 2d | 12-15 hours | Week 4 | Week 4 |
| Phase 2e | 10-12 hours | Week 5 | Week 5 |
| **Total** | **77-95 hours** | | **~5 weeks** |

### Minimum Viable Phase 2 (MVP)

If time is limited, implement only:
- Phase 2a: Smart Data Cleaning (20-25 hours)
- Phase 2b: Feature Engineering (20-25 hours)

**MVP Timeline:** 40-50 hours (~2-3 weeks)

---

## Success Metrics

### Phase 2a Success Criteria
- [ ] Detects 10+ types of data quality issues
- [ ] Recommendations are accurate (>90% user approval rate)
- [ ] Preview system works reliably
- [ ] Undo functionality is robust
- [ ] Processing time < 5 seconds for typical datasets

### Phase 2b Success Criteria
- [ ] Suggests 5-10 relevant features per dataset
- [ ] Generated code executes without errors
- [ ] Features improve model performance (measurable)
- [ ] Explanations are clear and actionable

### Phase 2c Success Criteria
- [ ] Explanations are insightful and accurate
- [ ] Users understand outlier context better
- [ ] Recommendations are followed >70% of the time

### Phase 2d Success Criteria
- [ ] Reports are professional and comprehensive
- [ ] Export works in all formats
- [ ] Generation time < 10 seconds

### Phase 2e Success Criteria
- [ ] Explanations are accurate and helpful
- [ ] Users learn from explanations
- [ ] Code quality improves

---

## Risk Mitigation

### Technical Risks

1. **Risk:** LLM generates incorrect recommendations
   - **Mitigation:** Validate all suggestions, allow user override, clear confidence scores

2. **Risk:** Performance issues with large datasets
   - **Mitigation:** Sampling strategy, progressive loading, caching

3. **Risk:** Complex edge cases in data cleaning
   - **Mitigation:** Extensive testing, safe defaults, manual override options

4. **Risk:** Memory constraints (16GB requirement)
   - **Mitigation:** Document requirements, optimize prompts, smaller models for specific tasks

### User Experience Risks

1. **Risk:** Users don't trust AI recommendations
   - **Mitigation:** Clear explanations, confidence scores, preview before apply

2. **Risk:** Too many recommendations overwhelm users
   - **Mitigation:** Priority sorting, progressive disclosure, batch operations

3. **Risk:** Undo system fails
   - **Mitigation:** Robust testing, multiple backup strategies, export before changes

---

## Next Steps

### Immediate Actions (Phase 2a)

1. ✅ Create this plan document
2. ⏳ Create `modules/ai/data_cleaning_advisor.py` skeleton
3. ⏳ Implement issue detection system
4. ⏳ Build recommendation engine
5. ⏳ Create UI
6. ⏳ Test and iterate

### Weekly Reviews

- Review progress every Friday
- Update this document with actuals
- Adjust timeline as needed
- Celebrate wins! 🎉

---

## Appendix

### Code Architecture

```
modules/ai/
├── __init__.py
├── model_manager.py (Phase 0)
├── llm_integration.py (Phase 0)
├── context_builder.py (Phase 0)
├── prompts.py (Phase 0)
├── ui_components.py (Phase 0)
├── chat_assistant.py (Phase 1a)
├── insights_generator.py (Phase 1b)
├── nl_query_translator.py (Phase 1c)
├── data_cleaning_advisor.py (Phase 2a) ← NEW
├── feature_engineering_advisor.py (Phase 2b) ← NEW
├── anomaly_explainer.py (Phase 2c) ← NEW
├── report_generator.py (Phase 2d) ← NEW
└── code_explainer.py (Phase 2e) ← NEW
```

### Dependencies

**New Libraries Needed:**
- `python-Levenshtein` - Fuzzy string matching for duplicates
- `reportlab` or `weasyprint` - PDF generation
- `python-docx` - Word document generation

**Add to requirements.txt:**
```
python-Levenshtein>=0.12.2
reportlab>=3.6.0
python-docx>=0.8.11
```

---

## Document Updates

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| Nov 7, 2024 | 1.0 | Initial creation | Claude Code |

---

**End of Phase 2 Plan**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
