# NLP-Driven Coursera Course Recommender System

A comprehensive content-based recommendation system for Coursera courses using Natural Language Processing techniques.

## 📁 Project Structure

```
NLP_Coursera_Recommender/
├── src/                                    # Source code
│   ├── coursera_recommender.py            # Core recommendation engine
│   ├── streamlit_app.py                   # Web interface
│   ├── evaluation_metrics.py              # Quality assessment framework
│   └── generate_nlp_recommender_report.py # Analysis report generator
├── reports/                               # Generated analysis reports
│   └── NLP_Recommender_Analysis_Report_*.txt
├── visualizations/                        # Generated charts and plots
│   ├── category_distribution.png
│   ├── rating_distribution.png
│   ├── similarity_heatmap.png
│   └── topic_distribution.png
├── docs/                                  # Documentation
│   ├── README.md                         # Detailed technical documentation
│   └── NLP_Recommender_Project_Summary.md # Project overview
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd NLP_Coursera_Recommender
pip install -r requirements.txt
```

### 2. Run the Core System Demo
```bash
python src/coursera_recommender.py
```

### 3. Launch Interactive Web Interface
```bash
streamlit run src/streamlit_app.py
```

### 4. Generate Analysis Report
```bash
python src/generate_nlp_recommender_report.py
```

### 5. Run Evaluation Metrics
```bash
python src/evaluation_metrics.py
```

## 🎯 Key Features

- **Content-Based Filtering**: Uses course descriptions, skills, and metadata
- **Dual NLP Approaches**: TF-IDF vectorization and Topic Modeling (LDA)
- **Interactive Web UI**: Streamlit-based interface with visualizations
- **Comprehensive Evaluation**: Multiple quality metrics and method comparison
- **Interest-Based Search**: Find courses by user-specified keywords

## 📊 System Performance

- **Dataset**: 10 sample Coursera courses across 3 categories
- **Coverage**: 90% catalog coverage
- **Speed**: Sub-second recommendation generation
- **Accuracy**: High similarity scoring with diversity optimization

## 📈 Results Summary

| Method | Coverage | Diversity | Popularity Bias | Intra-list Similarity |
|--------|----------|-----------|----------------|---------------------|
| TF-IDF | 0.900 | 0.633 | -0.030 | 0.103 |
| Topic Modeling | 0.900 | 0.683 | 0.060 | 0.059 |

**Winner**: Topic Modeling for diversity, TF-IDF for bias control

## 🔧 Technical Stack

- **Python 3.x** with pandas, numpy, scikit-learn
- **NLP**: NLTK for text preprocessing
- **ML**: TF-IDF, Latent Dirichlet Allocation, Cosine Similarity
- **UI**: Streamlit with Plotly visualizations
- **Evaluation**: Custom metrics framework

## 📋 Usage Examples

### Basic Recommendation
```python
from src.coursera_recommender import CourseraRecommender

recommender = CourseraRecommender()
recommender.create_sample_data()
recommender.build_tfidf_features()

# Get recommendations for a course
recs = recommender.get_course_recommendations('CS001', n_recommendations=5)
print(recs)
```

### Interest-Based Search
```python
# Search by interests
interests = ['machine learning', 'data analysis', 'python']
recs = recommender.get_recommendations_by_interests(interests)
print(recs)
```

## 📖 Documentation

- **Technical Details**: See `docs/README.md`
- **Project Summary**: See `docs/NLP_Recommender_Project_Summary.md`
- **Analysis Report**: See `reports/NLP_Recommender_Analysis_Report_*.txt`

## 🚧 Future Enhancements

- Larger course dataset integration
- Advanced NLP with transformers (BERT, RoBERTa)
- User personalization and learning paths
- Hybrid collaborative filtering
- Real-time course updates

## 📞 Support

For questions or issues, please refer to the documentation in the `docs/` directory or examine the source code in `src/`.

---

**Created**: September 2025
**Status**: Complete and functional
**License**: Educational/Demonstration purposes