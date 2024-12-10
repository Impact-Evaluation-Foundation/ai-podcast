import os
import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import re
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session
import argparse

# Load environment variables
load_dotenv()

CONSUMER_KEY = os.getenv("MIRAHEZE_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("MIRAHEZE_CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("MIRAHEZE_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("MIRAHEZE_ACCESS_SECRET")

API_URL = "https://ief.wiki/w/api.php"

def create_oauth_session():
    """Create an OAuth1 session using the access token and secret."""
    return OAuth1Session(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=ACCESS_TOKEN,
        resource_owner_secret=ACCESS_SECRET
    )

def get_csrf_token(oauth_session):
    """Fetch the CSRF token required for editing a wiki page."""
    params = {
        "action": "query",
        "meta": "tokens",
        "format": "json"
    }

    headers = {
        "User-Agent": "IERetrv/1.0 (impactevaluationfoundation@gmail.com)"
    }

    response = oauth_session.get(API_URL, params=params, headers=headers)
    
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")

    response.raise_for_status()
    
    data = response.json()
    return data["query"]["tokens"]["csrftoken"]

def create_wiki_page(title, content, oauth_session):
    """Create or edit a wiki page with authentication."""
    
    csrf_token = get_csrf_token(oauth_session)

    data = {
        "action": "edit",
        "title": title,
        "text": content,
        "summary": "Creating an IEF Article automatically",
        "format": "json",
        "token": csrf_token
    }
    
    headers = {
        "User-Agent": "IERetrv/1.0 (impactevaluationfoundation@gmail.com)"
    }

    try:
        response = oauth_session.post(API_URL, data=data, headers=headers)
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            print(f"Error creating wiki page for {title}: {result['error']['info']}")
        elif "edit" in result and result["edit"]["result"] == "Success":
            print(f"Successfully created page: {title}")
        else:
            print(f"Unexpected response for {title}: {result}")

    except requests.exceptions.RequestException as e:
        print(f"Error creating wiki page for {title}: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")

def sanitize_title(title):
    """Sanitize the title to remove special characters."""
    return re.sub(r'[<>:"/\\|?* ]', "_", title)

def main():
    parser = argparse.ArgumentParser(
        description="Upload Wikipedia-style articles to Miraheze wiki based on project information in an XLSX file."
    )

    parser.add_argument(
        "--projects-file",
        type=str,
        default="resources/projects.xlsx",
        help="Path to the XLSX file containing project names. Default is 'resources/projects.xlsx'."
    )

    args = parser.parse_args()

    # Read the XLSX file
    df = pd.read_excel(args.projects_file)

    if df.empty:
        print("The XLSX file contains no data.")
        return

    # Create OAuth session for subsequent requests
    oauth_session = create_oauth_session()

    # Iterate through each project
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading projects", leave=False):
        project_name = row["Name"]

        # Construct the file path for the project article
        article_file_path = f"articles/{sanitize_title(project_name)}.wiki"
        
        if os.path.exists(article_file_path):
            with open(article_file_path, "r", encoding="utf-8") as f:
                article_content = f.read()

            # Create the wiki page
            create_wiki_page(f"Articles:{sanitize_title(project_name)}", article_content, oauth_session)

            # Small delay to avoid rate limits
            time.sleep(1)
        else:
            print(f"Article file for {project_name} not found. Skipping...")

if __name__ == "__main__":
    main()
