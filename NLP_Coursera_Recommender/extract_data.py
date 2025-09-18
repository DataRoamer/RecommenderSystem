#!/usr/bin/env python3
"""Extract sample data from CourseraRecommender and save as CSV."""

import pandas as pd
import os
from src.coursera_recommender import CourseraRecommender

def main():
    # Initialize recommender and create sample data
    recommender = CourseraRecommender()
    courses_df = recommender.create_sample_data()

    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(data_dir, 'coursera_courses.csv')
    courses_df.to_csv(csv_path, index=False)

    print(f"Dataset saved to: {csv_path}")
    print(f"Number of courses: {len(courses_df)}")
    print(f"Columns: {list(courses_df.columns)}")

if __name__ == "__main__":
    main()