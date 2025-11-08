"""
Simple launcher script for the EDA Tool
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting EDA Tool...")

    # Change to the app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)

    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        print("Make sure you've installed the requirements:")
        print("pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

if __name__ == "__main__":
    main()