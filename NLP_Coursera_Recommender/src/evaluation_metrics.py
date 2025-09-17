import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    def __init__(self, recommender):
        """Initialize the evaluation module with a trained recommender."""
        self.recommender = recommender
        self.evaluation_results = {}

    def calculate_diversity(self, recommendations):
        """Calculate diversity of recommendations based on categories and levels."""
        if isinstance(recommendations, pd.DataFrame):
            categories = recommendations['category'].nunique()
            levels = recommendations['level'].nunique()
            total_recs = len(recommendations)

            category_diversity = categories / min(total_recs, self.recommender.courses_df['category'].nunique())
            level_diversity = levels / min(total_recs, self.recommender.courses_df['level'].nunique())

            return {
                'category_diversity': category_diversity,
                'level_diversity': level_diversity,
                'overall_diversity': (category_diversity + level_diversity) / 2
            }
        return {'category_diversity': 0, 'level_diversity': 0, 'overall_diversity': 0}

    def calculate_coverage(self, all_recommendations):
        """Calculate catalog coverage - how many unique courses are recommended."""
        unique_courses = set()
        total_courses = len(self.recommender.courses_df)

        for recs in all_recommendations:
            if isinstance(recs, pd.DataFrame):
                unique_courses.update(recs['course_id'].tolist())

        coverage = len(unique_courses) / total_courses
        return {
            'coverage': coverage,
            'unique_courses_recommended': len(unique_courses),
            'total_courses': total_courses
        }

    def calculate_popularity_bias(self, recommendations):
        """Calculate popularity bias in recommendations."""
        if isinstance(recommendations, pd.DataFrame):
            rec_ratings = []
            for _, rec in recommendations.iterrows():
                course_info = self.recommender.courses_df[
                    self.recommender.courses_df['course_id'] == rec['course_id']
                ]
                if not course_info.empty:
                    rec_ratings.append(course_info.iloc[0]['rating'])

            if rec_ratings:
                avg_rec_rating = np.mean(rec_ratings)
                overall_avg_rating = self.recommender.courses_df['rating'].mean()

                bias = avg_rec_rating - overall_avg_rating
                return {
                    'avg_recommended_rating': avg_rec_rating,
                    'overall_avg_rating': overall_avg_rating,
                    'popularity_bias': bias
                }

        return {'avg_recommended_rating': 0, 'overall_avg_rating': 0, 'popularity_bias': 0}

    def calculate_intra_list_similarity(self, recommendations):
        """Calculate similarity within a recommendation list."""
        if not isinstance(recommendations, pd.DataFrame) or len(recommendations) < 2:
            return {'intra_list_similarity': 0}

        # Get course indices
        course_indices = []
        for _, rec in recommendations.iterrows():
            course_idx = self.recommender.courses_df[
                self.recommender.courses_df['course_id'] == rec['course_id']
            ].index
            if len(course_idx) > 0:
                course_indices.append(course_idx[0])

        if len(course_indices) < 2 or self.recommender.similarity_matrix is None:
            return {'intra_list_similarity': 0}

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(course_indices)):
            for j in range(i + 1, len(course_indices)):
                sim = self.recommender.similarity_matrix[course_indices[i]][course_indices[j]]
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        return {
            'intra_list_similarity': avg_similarity,
            'num_pairs': len(similarities)
        }

    def evaluate_single_recommendation(self, course_id, n_recommendations=5, method='tfidf'):
        """Evaluate recommendations for a single course."""
        recommendations = self.recommender.get_course_recommendations(
            course_id, n_recommendations, method
        )

        if isinstance(recommendations, str):  # Error case
            return {'error': recommendations}

        # Calculate metrics
        diversity = self.calculate_diversity(recommendations)
        popularity_bias = self.calculate_popularity_bias(recommendations)
        intra_similarity = self.calculate_intra_list_similarity(recommendations)

        return {
            'course_id': course_id,
            'method': method,
            'num_recommendations': len(recommendations),
            'diversity': diversity,
            'popularity_bias': popularity_bias,
            'intra_similarity': intra_similarity,
            'recommendations': recommendations
        }

    def evaluate_all_courses(self, n_recommendations=5, methods=['tfidf', 'topic']):
        """Evaluate recommendations for all courses in the dataset."""
        results = []
        all_recommendations_by_method = defaultdict(list)

        for _, course in self.recommender.courses_df.iterrows():
            course_id = course['course_id']

            for method in methods:
                result = self.evaluate_single_recommendation(
                    course_id, n_recommendations, method
                )

                if 'error' not in result:
                    results.append(result)
                    all_recommendations_by_method[method].append(result['recommendations'])

        # Calculate overall metrics
        overall_metrics = {}
        for method in methods:
            if method in all_recommendations_by_method:
                # Coverage
                coverage = self.calculate_coverage(all_recommendations_by_method[method])

                # Average metrics
                method_results = [r for r in results if r['method'] == method]

                avg_diversity = np.mean([r['diversity']['overall_diversity'] for r in method_results])
                avg_popularity_bias = np.mean([r['popularity_bias']['popularity_bias'] for r in method_results])
                avg_intra_similarity = np.mean([r['intra_similarity']['intra_list_similarity'] for r in method_results])

                overall_metrics[method] = {
                    'coverage': coverage,
                    'avg_diversity': avg_diversity,
                    'avg_popularity_bias': avg_popularity_bias,
                    'avg_intra_similarity': avg_intra_similarity,
                    'num_evaluations': len(method_results)
                }

        self.evaluation_results = {
            'individual_results': results,
            'overall_metrics': overall_metrics
        }

        return self.evaluation_results

    def create_evaluation_report(self):
        """Create a comprehensive evaluation report."""
        if not self.evaluation_results:
            self.evaluate_all_courses()

        report = []
        report.append("=" * 80)
        report.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 80)

        overall_metrics = self.evaluation_results['overall_metrics']

        for method, metrics in overall_metrics.items():
            report.append(f"\n{method.upper()} METHOD EVALUATION:")
            report.append("-" * 50)
            report.append(f"Coverage: {metrics['coverage']['coverage']:.3f} ({metrics['coverage']['unique_courses_recommended']}/{metrics['coverage']['total_courses']} courses)")
            report.append(f"Average Diversity: {metrics['avg_diversity']:.3f}")
            report.append(f"Average Popularity Bias: {metrics['avg_popularity_bias']:.3f}")
            report.append(f"Average Intra-list Similarity: {metrics['avg_intra_similarity']:.3f}")
            report.append(f"Number of Evaluations: {metrics['num_evaluations']}")

        # Interpretation
        report.append("\n\nINTERPRETATION GUIDE:")
        report.append("-" * 50)
        report.append("Coverage: Higher is better (0-1 scale)")
        report.append("  - Measures how many unique courses are recommended across all evaluations")
        report.append("Diversity: Higher is better (0-1 scale)")
        report.append("  - Measures variety in categories and levels of recommended courses")
        report.append("Popularity Bias: Closer to 0 is better")
        report.append("  - Positive values indicate bias toward popular (high-rated) courses")
        report.append("  - Negative values indicate bias toward less popular courses")
        report.append("Intra-list Similarity: Lower is better (0-1 scale)")
        report.append("  - Measures how similar recommended courses are to each other")
        report.append("  - Lower values indicate more diverse recommendations")

        return "\n".join(report)

    def visualize_evaluation_results(self):
        """Create visualizations for evaluation results."""
        if not self.evaluation_results:
            self.evaluate_all_courses()

        overall_metrics = self.evaluation_results['overall_metrics']

        # Prepare data for plotting
        methods = list(overall_metrics.keys())
        coverage_scores = [overall_metrics[m]['coverage']['coverage'] for m in methods]
        diversity_scores = [overall_metrics[m]['avg_diversity'] for m in methods]
        bias_scores = [abs(overall_metrics[m]['avg_popularity_bias']) for m in methods]
        similarity_scores = [overall_metrics[m]['avg_intra_similarity'] for m in methods]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Recommendation System Evaluation Metrics', fontsize=16, fontweight='bold')

        # Coverage
        axes[0, 0].bar(methods, coverage_scores, color=['#3498db', '#e74c3c'])
        axes[0, 0].set_title('Coverage Score')
        axes[0, 0].set_ylabel('Coverage')
        axes[0, 0].set_ylim(0, 1)

        # Diversity
        axes[0, 1].bar(methods, diversity_scores, color=['#2ecc71', '#f39c12'])
        axes[0, 1].set_title('Average Diversity')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].set_ylim(0, 1)

        # Popularity Bias (absolute values)
        axes[1, 0].bar(methods, bias_scores, color=['#9b59b6', '#1abc9c'])
        axes[1, 0].set_title('Popularity Bias (Absolute)')
        axes[1, 0].set_ylabel('|Bias Score|')

        # Intra-list Similarity
        axes[1, 1].bar(methods, similarity_scores, color=['#e67e22', '#34495e'])
        axes[1, 1].set_title('Intra-list Similarity')
        axes[1, 1].set_ylabel('Similarity Score')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def compare_methods(self):
        """Compare different recommendation methods."""
        if not self.evaluation_results:
            self.evaluate_all_courses()

        overall_metrics = self.evaluation_results['overall_metrics']

        comparison_df = pd.DataFrame(overall_metrics).T
        comparison_df = comparison_df.round(3)

        # Create a summary comparison
        summary = []
        summary.append("METHOD COMPARISON SUMMARY")
        summary.append("=" * 50)

        methods = list(overall_metrics.keys())
        if len(methods) >= 2:
            method1, method2 = methods[0], methods[1]

            # Coverage comparison
            cov1 = overall_metrics[method1]['coverage']['coverage']
            cov2 = overall_metrics[method2]['coverage']['coverage']
            better_coverage = method1 if cov1 > cov2 else method2
            summary.append(f"Coverage: {better_coverage} performs better ({cov1:.3f} vs {cov2:.3f})")

            # Diversity comparison
            div1 = overall_metrics[method1]['avg_diversity']
            div2 = overall_metrics[method2]['avg_diversity']
            better_diversity = method1 if div1 > div2 else method2
            summary.append(f"Diversity: {better_diversity} performs better ({div1:.3f} vs {div2:.3f})")

            # Bias comparison
            bias1 = abs(overall_metrics[method1]['avg_popularity_bias'])
            bias2 = abs(overall_metrics[method2]['avg_popularity_bias'])
            better_bias = method1 if bias1 < bias2 else method2
            summary.append(f"Bias Control: {better_bias} performs better (|{bias1:.3f}| vs |{bias2:.3f}|)")

        return comparison_df, "\n".join(summary)

def main():
    """Demonstrate the evaluation system."""
    print("Recommendation System Evaluation")
    print("=" * 60)

    # Import and initialize recommender
    from coursera_recommender import CourseraRecommender

    recommender = CourseraRecommender()
    recommender.create_sample_data()
    recommender.build_tfidf_features()
    recommender.build_topic_features()

    # Initialize evaluator
    evaluator = RecommendationEvaluator(recommender)

    print("Running comprehensive evaluation...")

    # Single course evaluation example
    print("\nSingle Course Evaluation Example:")
    single_result = evaluator.evaluate_single_recommendation('CS001', n_recommendations=3)
    print(f"Course: {single_result['course_id']}")
    print(f"Diversity: {single_result['diversity']['overall_diversity']:.3f}")
    print(f"Popularity Bias: {single_result['popularity_bias']['popularity_bias']:.3f}")

    # Full evaluation
    print("\nRunning full evaluation (this may take a moment)...")
    results = evaluator.evaluate_all_courses(n_recommendations=3)

    # Generate report
    print("\nEvaluation Report:")
    report = evaluator.create_evaluation_report()
    print(report)

    # Method comparison
    print("\n\nMethod Comparison:")
    comparison_df, comparison_summary = evaluator.compare_methods()
    print(comparison_summary)

    print("\nDetailed Metrics Comparison:")
    print(comparison_df.to_string())

    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()