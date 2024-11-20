import pandas as pd
import argparse
from tqdm import tqdm
import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize API clients
API_KEY = os.getenv("PERPLEXITY_KEY")
researchClient = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")
aiClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create necessary directories
os.makedirs("research", exist_ok=True)
os.makedirs("articles", exist_ok=True)

# Utility: Log messages with formatting
def log_message(message):
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")

# Utility: Save data to a file
def save_to_file(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)
    print(f"Saved content to {file_path}")

# Utility: Sanitize filenames by removing special characters
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)

# Async function to fetch research data from Perplexity API
async def fetch_research_data(project_name):
    log_message(f"Fetching research data for project: {project_name}")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant specializing in retrieving detailed research "
                "information about projects based on a given name and URL. Your task "
                "is to gather data that will be used to draft a structured Wikipedia article."
            ),
        },
        {
            "role": "user",
            "content": f"""
        Collect detailed and well-structured information about the '{project_name}' project/entity. 
        Your task is to retrieve and summarize data to fit the structure of a Wikipedia article.

        Use the following structure as a guide:
        1. **Introduction**:
            - Provide an overview of the project/entity.
            - Explain its significance or purpose.
        2. **Role Summary** (Optional):
            - Include expanded details about its scope or mission.
        3. **Key Features/Responsibilities** (Optional):
            - Describe its functions or unique aspects.
        4. **Skills and Qualifications** (Optional):
            - Include any qualifications or certifications it involves.
        5. **Methodologies and Tools** (Optional):
            - Highlight methodologies or tools associated with it.
        6. **Impact/Significance** (Optional):
            - Discuss societal, environmental, or economic contributions.
        7. **Case Studies and Examples** (Optional):
            - Provide real-world examples to illustrate its impact.
        8. **See Also** (Optional):
            - Suggest related topics for further exploration.
        9. **References**:
            - Include URLs in full text (with access dates if possible).
        """,
        },
    ]

    response = researchClient.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    return response.choices[0].message.content

# Async function to generate an article using ChatGPT based on research data
async def generate_article_from_research(project_name, research_data):
    log_message(f"Generating Wikipedia-style article for project: {project_name}")
    messages = [
        {
            "role": "system",
            "content": (
                """
You are an expert at writing Wikipedia articles that are both informative and engaging. When generating content, ensure it adheres to Wikipedia's neutral tone and encyclopedic style. The output must be written in Wikimedia markup (not Markdown) and formatted precisely.

Your task is to transform provided research into a structured Wikipedia article. The article should flow naturally, be concise, and avoid technical jargon where possible, while remaining accurate and detailed.

Requirements:
- **Output in Wikimedia markup** only (no Markdown or plain text formatting).
- **References must always be included**, even if the content approaches the token limit.
  - Include references at the end of the article as a numbered list in proper Wikimedia markup using full URLs and access dates (if available).
  - If necessary, truncate the article's body content slightly to prioritize space for references.
- Avoid adding internal Wikipedia links (e.g., to related articles) for now.
- External references, URLs, or sources must be properly formatted in the references section.
- If data for a specific section is missing, omit the section header entirely.

The final output must be concise yet comprehensive and adhere strictly to Wikimedia standards.
                """
            ),
        },
        {
            "role": "user",
            "content": f"""
Using the following research data, generate a structured Wikipedia article for the '{project_name}' project/entity. 

### Research Data:
{research_data}

Key Points to Remember:
1. Write the article in a structured format directly in Wikimedia markup (do not use Markdown or plain text).
2. Include all references at the end of the article in a numbered list with proper Wikimedia markup.
3. Do not include a section header if no data exists for that section.
4. If token limits are reached, truncate non-critical sections, but always include all references.
        """,
        },
    ]

    response = aiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return response.choices[0].message.content

# Async function to process each project
async def process_projects(projects_file_path):
    try:
        # Load the XLSX file
        data = pd.read_excel(projects_file_path)

        if data.empty:
            log_message("The file contains no data.")
            return

        # Add "Article" column if it doesn't exist
        if "Article" not in data.columns:
            data["Article"] = ""

        log_message("Processing projects and generating research and articles:")

        for index, row in tqdm(
            data.iterrows(), total=data.shape[0], unit="project", colour="yellow"
        ):
            project_name = row.get("Name", f"Project_{index+1}")

            # Fetch research data
            research_data = await fetch_research_data(project_name)

            # Sanitize project name for filenames
            sanitized_project_name = sanitize_filename(project_name)

            # Save research data
            research_file_path = f"research/{sanitized_project_name}.txt"
            save_to_file(research_file_path, research_data)

            # Generate Wikipedia-style article
            article_content = await generate_article_from_research(
                project_name, research_data
            )

            # Save article content to a .wiki file
            article_file_path = f"articles/{sanitized_project_name}.wiki"
            save_to_file(article_file_path, article_content)

            # Update the "Article" column in the XLSX file (might fail since articles are quite big)
            data.at[index, "Article"] = article_content

        # Save the updated XLSX file
        data.to_excel(projects_file_path, index=False)
        log_message("Processing completed, XLSX file updated with generated articles.")

    except Exception as e:
        log_message(f"Error: {e}")

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Generate research and Wikipedia-style articles for projects in an XLSX file."
    )

    parser.add_argument(
        "--file-path",
        type=str,
        default="resources/projects.xlsx",
        help='Path to the XLSX file. Default is "resources/projects.xlsx".',
    )

    args = parser.parse_args()

    asyncio.run(process_projects(args.file_path))

if __name__ == "__main__":
    main()
