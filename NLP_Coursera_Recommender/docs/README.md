# NLP-Driven Content-Based Coursera Recommender System

A comprehensive content-based recommendation system for Coursera courses that leverages Natural Language Processing (NLP) techniques to provide personalized course recommendations.

## Features

- **Content-Based Filtering**: Uses course descriptions, skills, and metadata to recommend similar courses
- **Multiple NLP Approaches**:
  - TF-IDF vectorization for keyword-based similarity
  - Topic modeling using Latent Dirichlet Allocation (LDA)
- **Interactive Web Interface**: Streamlit-based UI for easy interaction
- **Comprehensive Evaluation**: Multiple metrics to assess recommendation quality
- **Interest-Based Search**: Find courses based on user-specified interests

## System Architecture

### Core Components

1. **CourseraRecommender** (`coursera_recommender.py`)
   - Main recommendation engine
   - Text preprocessing pipeline
   - TF-IDF and LDA feature extraction
   - Similarity calculation and recommendation generation

2. **Streamlit Web Interface** (`streamlit_app.py`)
   - Interactive course browsing and recommendation
   - Visualization of similarity scores and topic distributions
   - Multiple recommendation pages and analysis tools

3. **Evaluation System** (`evaluation_metrics.py`)
   - Recommendation quality assessment
   - Diversity, coverage, and bias metrics
   - Method comparison and reporting

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main demonstration:
```bash
python coursera_recommender.py
```

3. Launch the web interface:
```bash
streamlit run streamlit_app.py
```

## Usage Examples

### Basic Recommendation
```python
from coursera_recommender import CourseraRecommender

# Initialize and load data
recommender = CourseraRecommender()
recommender.create_sample_data()
recommender.build_tfidf_features()

# Get recommendations for a course
recommendations = recommender.get_course_recommendations('CS001', n_recommendations=5)
print(recommendations)
```

### Interest-Based Search
```python
# Search by interests
interests = ['machine learning', 'data analysis', 'python']
recommendations = recommender.get_recommendations_by_interests(interests)
print(recommendations)
```

### Evaluation
```python
from evaluation_metrics import RecommendationEvaluator

evaluator = RecommendationEvaluator(recommender)
results = evaluator.evaluate_all_courses()
report = evaluator.create_evaluation_report()
print(report)
```

## Web Interface Features

The Streamlit web application provides four main pages:

1. **Course Recommendations**: Get recommendations based on selected courses
2. **Interest-Based Search**: Find courses using keyword interests
3. **Course Analysis**: Detailed analysis of individual courses including topic distribution and keyword importance
4. **Dataset Overview**: Statistics and visualizations of the course dataset

## Dataset

The system includes a sample dataset of 10 Coursera courses across different categories:
- Computer Science (6 courses)
- Data Science (3 courses)
- Business (2 courses)

Each course includes:
- Title and description
- Skills taught
- Difficulty level
- University/provider
- Rating and duration

## NLP Processing Pipeline

1. **Text Preprocessing**:
   - Lowercasing
   - Special character removal
   - Tokenization (NLTK)
   - Stop word removal
   - Lemmatization

2. **Feature Extraction**:
   - TF-IDF vectorization with n-grams (1-2)
   - Topic modeling with LDA (5 topics)
   - Cosine similarity calculation

3. **Recommendation Generation**:
   - Course-to-course similarity
   - Interest matching
   - Configurable number of recommendations

## Evaluation Metrics

The system includes comprehensive evaluation metrics:

- **Coverage**: Proportion of unique courses recommended
- **Diversity**: Variety in categories and difficulty levels
- **Popularity Bias**: Tendency to recommend popular courses
- **Intra-list Similarity**: Similarity within recommendation lists

## Method Comparison

Based on evaluation results:

| Metric | TF-IDF | Topic Modeling |
|--------|--------|----------------|
| Coverage | 0.900 | 0.900 |
| Diversity | 0.633 | 0.683 |
| Popularity Bias | -0.030 | 0.060 |
| Intra-list Similarity | 0.103 | 0.059 |

**Topic modeling** performs better for diversity and produces less similar recommendations, while **TF-IDF** has better bias control.

## File Structure

```
├── coursera_recommender.py    # Main recommendation system
├── streamlit_app.py          # Web interface
├── evaluation_metrics.py     # Evaluation framework
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Technical Details

### Dependencies
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: ML algorithms (TF-IDF, LDA, cosine similarity)
- nltk: Natural language processing
- streamlit: Web interface
- plotly: Interactive visualizations

### Performance
- Processes 10 courses in under 1 second
- Scalable to larger datasets
- Memory-efficient similarity calculations

## Future Enhancements

Potential improvements for the system:

1. **Larger Dataset**: Integration with real Coursera course data
2. **Advanced NLP**:
   - Word embeddings (Word2Vec, GloVe)
   - Transformer models (BERT, RoBERTa)
   - Neural topic models
3. **Hybrid Approaches**: Combine content-based with collaborative filtering
4. **User Profiles**: Personalized recommendations based on learning history
5. **Real-time Updates**: Dynamic course catalog updates
6. **A/B Testing**: Framework for testing different recommendation strategies

## License

This project is created for educational and demonstration purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.