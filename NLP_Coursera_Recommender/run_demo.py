#!/usr/bin/env python3
"""
Quick launcher script for the NLP-Driven Coursera Recommender System
"""

import os
import sys
import subprocess

def main():
    print("🎓 NLP-Driven Coursera Recommender System")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run core system demo")
    print("2. Launch web interface")
    print("3. Generate analysis report")
    print("4. Run evaluation metrics")
    print("5. Exit")

    while True:
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            print("\n🔧 Running core system demo...")
            subprocess.run([sys.executable, "src/coursera_recommender.py"])

        elif choice == '2':
            print("\n🌐 Launching web interface...")
            print("This will open in your browser at http://localhost:8501")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])

        elif choice == '3':
            print("\n📊 Generating analysis report...")
            subprocess.run([sys.executable, "src/generate_nlp_recommender_report.py"])

        elif choice == '4':
            print("\n📈 Running evaluation metrics...")
            subprocess.run([sys.executable, "src/evaluation_metrics.py"])

        elif choice == '5':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1-5.")
            continue

        print("\n" + "=" * 50)
        print("Choose another option or press 5 to exit:")

if __name__ == "__main__":
    main()