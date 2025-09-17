import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

class CourseraRecommender:
    def __init__(self):
        """Initialize the NLP-driven content-based recommender system."""
        self.tfidf_vectorizer = None
        self.course_features = None
        self.similarity_matrix = None
        self.courses_df = None
        self.lda_model = None
        self.topic_features = None

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def create_sample_data(self):
        """Create sample Coursera course dataset for demonstration."""
        courses_data = [
            {
                'course_id': 'CS001',
                'title': 'Machine Learning Fundamentals',
                'description': 'Learn the basics of machine learning algorithms including linear regression, decision trees, and neural networks. This course covers supervised and unsupervised learning techniques with practical Python implementations.',
                'skills': 'Python, Scikit-learn, Data Analysis, Statistics',
                'level': 'Beginner',
                'category': 'Computer Science',
                'university': 'Stanford University',
                'rating': 4.7,
                'duration': '6 weeks'
            },
            {
                'course_id': 'CS002',
                'title': 'Deep Learning Specialization',
                'description': 'Master deep learning and neural networks. Build convolutional neural networks for computer vision, recurrent neural networks for sequence modeling, and learn about transformers and attention mechanisms.',
                'skills': 'TensorFlow, Keras, Computer Vision, NLP',
                'level': 'Advanced',
                'category': 'Computer Science',
                'university': 'DeepLearning.AI',
                'rating': 4.9,
                'duration': '12 weeks'
            },
            {
                'course_id': 'DS001',
                'title': 'Data Science Methodology',
                'description': 'Learn the data science pipeline from data collection to model deployment. Cover data cleaning, exploratory data analysis, feature engineering, and statistical modeling techniques.',
                'skills': 'Data Analysis, Statistics, R, Python',
                'level': 'Intermediate',
                'category': 'Data Science',
                'university': 'IBM',
                'rating': 4.5,
                'duration': '8 weeks'
            },
            {
                'course_id': 'BZ001',
                'title': 'Digital Marketing Analytics',
                'description': 'Understand digital marketing metrics, customer segmentation, and campaign optimization. Learn to use analytics tools for measuring marketing effectiveness and ROI.',
                'skills': 'Google Analytics, Marketing, Data Visualization',
                'level': 'Beginner',
                'category': 'Business',
                'university': 'University of Illinois',
                'rating': 4.3,
                'duration': '4 weeks'
            },
            {
                'course_id': 'CS003',
                'title': 'Natural Language Processing',
                'description': 'Explore text processing, sentiment analysis, named entity recognition, and language modeling. Build chatbots and text classification systems using modern NLP techniques.',
                'skills': 'NLTK, spaCy, Text Mining, Python',
                'level': 'Advanced',
                'category': 'Computer Science',
                'university': 'University of Michigan',
                'rating': 4.6,
                'duration': '10 weeks'
            },
            {
                'course_id': 'DS002',
                'title': 'Data Visualization with Tableau',
                'description': 'Create compelling data visualizations and dashboards using Tableau. Learn design principles, interactive visualization techniques, and storytelling with data.',
                'skills': 'Tableau, Data Visualization, Dashboard Design',
                'level': 'Beginner',
                'category': 'Data Science',
                'university': 'University of California Davis',
                'rating': 4.4,
                'duration': '5 weeks'
            },
            {
                'course_id': 'CS004',
                'title': 'Algorithms and Data Structures',
                'description': 'Master fundamental algorithms and data structures including sorting, searching, graph algorithms, dynamic programming, and complexity analysis.',
                'skills': 'Algorithms, Data Structures, Problem Solving, Java',
                'level': 'Intermediate',
                'category': 'Computer Science',
                'university': 'Princeton University',
                'rating': 4.8,
                'duration': '7 weeks'
            },
            {
                'course_id': 'BZ002',
                'title': 'Financial Markets and Investment',
                'description': 'Learn about financial markets, investment strategies, portfolio management, and risk assessment. Understand stocks, bonds, derivatives, and market analysis.',
                'skills': 'Finance, Investment Analysis, Risk Management',
                'level': 'Intermediate',
                'category': 'Business',
                'university': 'Yale University',
                'rating': 4.7,
                'duration': '8 weeks'
            },
            {
                'course_id': 'DS003',
                'title': 'Statistical Analysis with R',
                'description': 'Learn statistical analysis using R programming. Cover hypothesis testing, regression analysis, ANOVA, and advanced statistical modeling techniques.',
                'skills': 'R Programming, Statistics, Data Analysis, Regression',
                'level': 'Intermediate',
                'category': 'Data Science',
                'university': 'Johns Hopkins University',
                'rating': 4.5,
                'duration': '6 weeks'
            },
            {
                'course_id': 'CS005',
                'title': 'Computer Vision Fundamentals',
                'description': 'Learn image processing, object detection, and computer vision algorithms. Build applications for image recognition, facial detection, and autonomous systems.',
                'skills': 'OpenCV, Image Processing, Python, Computer Vision',
                'level': 'Advanced',
                'category': 'Computer Science',
                'university': 'University of Buffalo',
                'rating': 4.6,
                'duration': '9 weeks'
            }
        ]

        self.courses_df = pd.DataFrame(courses_data)
        return self.courses_df

    def preprocess_text(self, text):
        """Clean and preprocess text data for NLP analysis."""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                processed_tokens.append(self.lemmatizer.lemmatize(token))

        return ' '.join(processed_tokens)

    def create_content_features(self):
        """Create combined content features from course information."""
        # Combine relevant text fields
        self.courses_df['combined_content'] = (
            self.courses_df['title'] + ' ' +
            self.courses_df['description'] + ' ' +
            self.courses_df['skills'] + ' ' +
            self.courses_df['category']
        )

        # Preprocess the combined content
        self.courses_df['processed_content'] = self.courses_df['combined_content'].apply(
            self.preprocess_text
        )

        return self.courses_df['processed_content']

    def build_tfidf_features(self, max_features=1000):
        """Build TF-IDF features from course content."""
        processed_content = self.create_content_features()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )

        self.course_features = self.tfidf_vectorizer.fit_transform(processed_content)

        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.course_features)

        return self.course_features

    def build_topic_features(self, n_topics=5):
        """Build topic modeling features using LDA."""
        if self.course_features is None:
            self.build_tfidf_features()

        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )

        self.topic_features = self.lda_model.fit_transform(self.course_features)

        return self.topic_features

    def get_course_recommendations(self, course_id, n_recommendations=5, method='tfidf'):
        """Get course recommendations based on content similarity."""
        if course_id not in self.courses_df['course_id'].values:
            return f"Course ID {course_id} not found."

        # Get course index
        course_idx = self.courses_df[self.courses_df['course_id'] == course_id].index[0]

        if method == 'tfidf':
            if self.similarity_matrix is None:
                self.build_tfidf_features()

            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[course_idx]))

        elif method == 'topic':
            if self.topic_features is None:
                self.build_topic_features()

            # Calculate cosine similarity based on topic features
            topic_similarity = cosine_similarity([self.topic_features[course_idx]],
                                               self.topic_features)[0]
            sim_scores = list(enumerate(topic_similarity))

        # Sort by similarity (excluding the course itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

        # Get top recommendations
        top_indices = [i[0] for i in sim_scores[:n_recommendations]]

        recommendations = self.courses_df.iloc[top_indices][
            ['course_id', 'title', 'category', 'level', 'rating', 'university']
        ].copy()

        recommendations['similarity_score'] = [sim_scores[i][1] for i in range(n_recommendations)]

        return recommendations

    def get_recommendations_by_interests(self, interests, n_recommendations=5):
        """Get course recommendations based on user interests/keywords."""
        if self.tfidf_vectorizer is None:
            self.build_tfidf_features()

        # Preprocess user interests
        processed_interests = self.preprocess_text(' '.join(interests))

        # Transform user interests using the fitted vectorizer
        interests_vector = self.tfidf_vectorizer.transform([processed_interests])

        # Calculate similarity with all courses
        similarity_scores = cosine_similarity(interests_vector, self.course_features)[0]

        # Get top recommendations
        top_indices = np.argsort(similarity_scores)[::-1][:n_recommendations]

        recommendations = self.courses_df.iloc[top_indices][
            ['course_id', 'title', 'category', 'level', 'rating', 'university']
        ].copy()

        recommendations['similarity_score'] = similarity_scores[top_indices]

        return recommendations

    def get_topic_distribution(self, course_id):
        """Get topic distribution for a specific course."""
        if self.topic_features is None:
            self.build_topic_features()

        if course_id not in self.courses_df['course_id'].values:
            return f"Course ID {course_id} not found."

        course_idx = self.courses_df[self.courses_df['course_id'] == course_id].index[0]
        topic_dist = self.topic_features[course_idx]

        return {f'Topic_{i+1}': round(prob, 3) for i, prob in enumerate(topic_dist)}

    def get_feature_importance(self, course_id, top_n=10):
        """Get the most important TF-IDF features for a course."""
        if self.tfidf_vectorizer is None:
            self.build_tfidf_features()

        if course_id not in self.courses_df['course_id'].values:
            return f"Course ID {course_id} not found."

        course_idx = self.courses_df[self.courses_df['course_id'] == course_id].index[0]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        course_features_dense = self.course_features[course_idx].toarray()[0]

        # Get top features with their scores
        top_indices = np.argsort(course_features_dense)[::-1][:top_n]
        top_features = [(feature_names[i], course_features_dense[i])
                       for i in top_indices if course_features_dense[i] > 0]

        return top_features

    def display_course_info(self, course_id):
        """Display detailed information about a specific course."""
        if course_id not in self.courses_df['course_id'].values:
            return f"Course ID {course_id} not found."

        course_info = self.courses_df[self.courses_df['course_id'] == course_id].iloc[0]

        print(f"\n{'='*60}")
        print(f"COURSE INFORMATION")
        print(f"{'='*60}")
        print(f"Course ID: {course_info['course_id']}")
        print(f"Title: {course_info['title']}")
        print(f"Category: {course_info['category']}")
        print(f"Level: {course_info['level']}")
        print(f"University: {course_info['university']}")
        print(f"Rating: {course_info['rating']}/5.0")
        print(f"Duration: {course_info['duration']}")
        print(f"Skills: {course_info['skills']}")
        print(f"\nDescription:")
        print(f"{course_info['description']}")
        print(f"{'='*60}\n")

        return course_info

def main():
    """Demonstrate the NLP-driven content-based recommender system."""
    print("Coursera NLP-Driven Content-Based Recommender System")
    print("="*60)

    # Initialize recommender
    recommender = CourseraRecommender()

    # Create sample dataset
    print("Loading sample Coursera course dataset...")
    courses_df = recommender.create_sample_data()
    print(f"Loaded {len(courses_df)} courses")

    # Build features
    print("\nBuilding NLP features...")
    recommender.build_tfidf_features()
    recommender.build_topic_features()
    print("Features built successfully!")

    # Display available courses
    print("\nAvailable Courses:")
    print("-" * 60)
    for _, course in courses_df.iterrows():
        print(f"{course['course_id']}: {course['title']} ({course['category']})")

    # Example 1: Course-to-course recommendations
    print("\n\nExample 1: Course-to-Course Recommendations")
    print("="*60)
    target_course = 'CS001'
    recommender.display_course_info(target_course)

    print(f"Recommendations based on '{target_course}' (TF-IDF method):")
    recommendations = recommender.get_course_recommendations(target_course, n_recommendations=3)
    print(recommendations.to_string(index=False))

    print(f"\nRecommendations based on '{target_course}' (Topic modeling method):")
    topic_recommendations = recommender.get_course_recommendations(
        target_course, n_recommendations=3, method='topic'
    )
    print(topic_recommendations.to_string(index=False))

    # Example 2: Interest-based recommendations
    print("\n\nExample 2: Interest-Based Recommendations")
    print("="*60)
    user_interests = ['machine learning', 'data analysis', 'python programming']
    print(f"User interests: {user_interests}")

    interest_recommendations = recommender.get_recommendations_by_interests(
        user_interests, n_recommendations=4
    )
    print("\nRecommended courses based on interests:")
    print(interest_recommendations.to_string(index=False))

    # Example 3: Topic analysis
    print("\n\nExample 3: Topic Distribution Analysis")
    print("="*60)
    course_topics = recommender.get_topic_distribution('CS002')
    print(f"Topic distribution for course CS002:")
    for topic, prob in course_topics.items():
        print(f"  {topic}: {prob}")

    # Example 4: Feature importance
    print("\n\nExample 4: Important Keywords Analysis")
    print("="*60)
    important_features = recommender.get_feature_importance('CS003', top_n=8)
    print(f"Most important keywords for course CS003:")
    for feature, score in important_features:
        print(f"  {feature}: {score:.3f}")

    print("\nRecommendation system demonstration completed!")

if __name__ == "__main__":
    main()