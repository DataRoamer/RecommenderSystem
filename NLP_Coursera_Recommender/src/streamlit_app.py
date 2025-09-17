import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from coursera_recommender import CourseraRecommender
import numpy as np

# Configure page
st.set_page_config(
    page_title="Coursera Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .course-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load and initialize the recommender system."""
    recommender = CourseraRecommender()
    recommender.create_sample_data()
    recommender.build_tfidf_features()
    recommender.build_topic_features()
    return recommender

def display_course_card(course_row):
    """Display a course in a card format."""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {course_row['title']}")
            st.markdown(f"**University:** {course_row['university']}")
            st.markdown(f"**Category:** {course_row['category']} | **Level:** {course_row['level']}")
            if 'similarity_score' in course_row:
                st.markdown(f"**Similarity Score:** {course_row['similarity_score']:.3f}")

        with col2:
            st.metric("Rating", f"{course_row['rating']}/5.0")
            if 'duration' in course_row:
                st.markdown(f"**Duration:** {course_row['duration']}")

def create_similarity_heatmap(recommender, course_id):
    """Create a heatmap showing course similarities."""
    if recommender.similarity_matrix is None:
        return None

    course_idx = recommender.courses_df[recommender.courses_df['course_id'] == course_id].index[0]
    similarities = recommender.similarity_matrix[course_idx]

    course_titles = [title[:30] + "..." if len(title) > 30 else title
                    for title in recommender.courses_df['title']]

    fig = go.Figure(data=go.Heatmap(
        z=[similarities],
        x=course_titles,
        y=[recommender.courses_df.iloc[course_idx]['title'][:30]],
        colorscale='Blues',
        showscale=True
    ))

    fig.update_layout(
        title="Course Similarity Scores",
        xaxis_title="Courses",
        height=300
    )

    return fig

def create_topic_distribution_chart(recommender, course_id):
    """Create a bar chart showing topic distribution."""
    topic_dist = recommender.get_topic_distribution(course_id)

    if isinstance(topic_dist, str):
        return None

    topics = list(topic_dist.keys())
    probabilities = list(topic_dist.values())

    fig = px.bar(
        x=topics,
        y=probabilities,
        title=f"Topic Distribution for Course {course_id}",
        labels={'x': 'Topics', 'y': 'Probability'},
        color=probabilities,
        color_continuous_scale='Viridis'
    )

    return fig

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🎓 Coursera Course Recommender</h1>',
                unsafe_allow_html=True)
    st.markdown("### NLP-Driven Content-Based Recommendation System")

    # Load recommender
    with st.spinner("Loading recommendation system..."):
        recommender = load_recommender()

    # Sidebar
    st.sidebar.header("🔧 Recommendation Options")

    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Course Recommendations", "Interest-Based Search", "Course Analysis", "Dataset Overview"]
    )

    if page == "Course Recommendations":
        st.header("📚 Course-to-Course Recommendations")

        # Course selection
        course_options = [(row['course_id'], f"{row['course_id']}: {row['title']}")
                         for _, row in recommender.courses_df.iterrows()]

        selected_course = st.selectbox(
            "Select a course to get recommendations:",
            options=[option[0] for option in course_options],
            format_func=lambda x: next(option[1] for option in course_options if option[0] == x)
        )

        # Method selection
        method = st.sidebar.radio(
            "Recommendation Method:",
            ["TF-IDF", "Topic Modeling"],
            help="TF-IDF uses word importance, Topic Modeling uses thematic similarity"
        )

        # Number of recommendations
        n_recs = st.sidebar.slider("Number of recommendations:", 1, 8, 5)

        if selected_course:
            # Display selected course info
            st.subheader("📖 Selected Course")
            selected_course_info = recommender.courses_df[
                recommender.courses_df['course_id'] == selected_course
            ].iloc[0]

            display_course_card(selected_course_info)

            with st.expander("View Course Description"):
                st.write(selected_course_info['description'])
                st.write(f"**Skills:** {selected_course_info['skills']}")

            # Get recommendations
            method_param = 'tfidf' if method == 'TF-IDF' else 'topic'
            recommendations = recommender.get_course_recommendations(
                selected_course, n_recs, method_param
            )

            # Display recommendations
            st.subheader(f"🎯 Top {n_recs} Recommendations ({method})")

            for idx, (_, course) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"#### {idx}. Recommendation")
                display_course_card(course)

                # Get full course details
                full_course = recommender.courses_df[
                    recommender.courses_df['course_id'] == course['course_id']
                ].iloc[0]

                with st.expander(f"View details for {course['title']}"):
                    st.write(f"**Description:** {full_course['description']}")
                    st.write(f"**Skills:** {full_course['skills']}")

                st.markdown("---")

            # Visualization
            st.subheader("📊 Similarity Analysis")
            fig = create_similarity_heatmap(recommender, selected_course)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Interest-Based Search":
        st.header("🎯 Find Courses by Your Interests")

        # Interest input
        st.subheader("What are you interested in learning?")

        col1, col2 = st.columns([3, 1])

        with col1:
            interests_input = st.text_area(
                "Enter your interests (comma-separated):",
                placeholder="e.g., machine learning, data visualization, python programming",
                height=100
            )

        with col2:
            st.markdown("**Quick Tags:**")
            quick_tags = [
                "Machine Learning", "Data Science", "Python", "Statistics",
                "Business Analytics", "Computer Vision", "NLP", "Finance"
            ]

            selected_tags = []
            for tag in quick_tags:
                if st.checkbox(tag, key=f"tag_{tag}"):
                    selected_tags.append(tag)

        # Combine inputs
        all_interests = []
        if interests_input:
            all_interests.extend([interest.strip() for interest in interests_input.split(',')])
        all_interests.extend(selected_tags)

        n_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

        if all_interests and st.button("🔍 Get Recommendations"):
            with st.spinner("Finding courses for you..."):
                recommendations = recommender.get_recommendations_by_interests(
                    all_interests, n_recs
                )

            st.subheader(f"🎯 Top {n_recs} Courses for Your Interests")
            st.write(f"**Your interests:** {', '.join(all_interests)}")

            for idx, (_, course) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"#### {idx}. {course['title']}")
                display_course_card(course)

                # Get full course details
                full_course = recommender.courses_df[
                    recommender.courses_df['course_id'] == course['course_id']
                ].iloc[0]

                with st.expander(f"View details"):
                    st.write(f"**Description:** {full_course['description']}")
                    st.write(f"**Skills:** {full_course['skills']}")

                st.markdown("---")

    elif page == "Course Analysis":
        st.header("🔍 Course Analysis & Insights")

        # Course selection
        course_options = [(row['course_id'], f"{row['course_id']}: {row['title']}")
                         for _, row in recommender.courses_df.iterrows()]

        selected_course = st.selectbox(
            "Select a course to analyze:",
            options=[option[0] for option in course_options],
            format_func=lambda x: next(option[1] for option in course_options if option[0] == x)
        )

        if selected_course:
            # Course info
            course_info = recommender.courses_df[
                recommender.courses_df['course_id'] == selected_course
            ].iloc[0]

            st.subheader("📖 Course Information")
            display_course_card(course_info)

            col1, col2 = st.columns(2)

            with col1:
                # Topic distribution
                st.subheader("📊 Topic Distribution")
                fig_topics = create_topic_distribution_chart(recommender, selected_course)
                if fig_topics:
                    st.plotly_chart(fig_topics, use_container_width=True)

            with col2:
                # Important keywords
                st.subheader("🔑 Key Features")
                important_features = recommender.get_feature_importance(selected_course, top_n=10)

                if important_features:
                    keywords_df = pd.DataFrame(
                        important_features,
                        columns=['Keyword', 'Importance']
                    )

                    fig_keywords = px.bar(
                        keywords_df,
                        x='Importance',
                        y='Keyword',
                        orientation='h',
                        title="Most Important Keywords",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_keywords.update_layout(height=400)
                    st.plotly_chart(fig_keywords, use_container_width=True)

    elif page == "Dataset Overview":
        st.header("📊 Dataset Overview")

        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Courses", len(recommender.courses_df))

        with col2:
            st.metric("Categories", recommender.courses_df['category'].nunique())

        with col3:
            avg_rating = recommender.courses_df['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}/5.0")

        with col4:
            st.metric("Universities", recommender.courses_df['university'].nunique())

        # Category distribution
        st.subheader("📈 Course Distribution by Category")
        category_counts = recommender.courses_df['category'].value_counts()

        fig_category = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Courses by Category"
        )
        st.plotly_chart(fig_category, use_container_width=True)

        # Level distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Difficulty Level Distribution")
            level_counts = recommender.courses_df['level'].value_counts()
            fig_level = px.bar(
                x=level_counts.index,
                y=level_counts.values,
                title="Courses by Difficulty Level",
                color=level_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_level, use_container_width=True)

        with col2:
            st.subheader("⭐ Rating Distribution")
            fig_ratings = px.histogram(
                recommender.courses_df,
                x='rating',
                nbins=10,
                title="Course Ratings Distribution",
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

        # Course table
        st.subheader("📚 All Courses")
        st.dataframe(
            recommender.courses_df[['course_id', 'title', 'category', 'level', 'rating', 'university']],
            use_container_width=True
        )

if __name__ == "__main__":
    main()