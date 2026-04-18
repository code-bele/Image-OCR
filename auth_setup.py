import subprocess
import sys

def setup_gcloud():
    print("Checking for gcloud CLI...")
    try:
        # Check if gcloud is installed
        subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: gcloud CLI not found. Please install Google Cloud SDK first.")
        sys.exit(1)

    print("Please follow the instructions in your browser to login.")
    # This replaces 'auth.authenticate_user()' from Colab for local use
    subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
    print("Authentication successful!")

if __name__ == "__main__":
    setup_gcloud()