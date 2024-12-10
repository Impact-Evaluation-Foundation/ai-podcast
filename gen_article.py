import pandas as pd
import argparse
from tqdm import tqdm
import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
from gpt_researcher import GPTResearcher
import subprocess


# Load environment variables
load_dotenv()

# Initialize API clients
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY']=os.getenv("TAVILY_API_KEY")
API_KEY = os.getenv("PERPLEXITY_KEY")
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
    return re.sub(r'[<>:"/\\|?* ]', "_", name)

# Load prompts from prompts.xlsx
def load_prompts(prompts_file_path):
    return pd.read_excel(prompts_file_path)

# Async function to fetch research data with Tavily x GPT
async def fetch_research_data(project_name, system_prompt, user_prompt):
    log_message(f"Fetching research data for project: {project_name}")
    researcher = GPTResearcher(user_prompt, "research_report")
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    
    return report


# Async function to generate an article using a specific system prompt
async def generate_article_from_research(project_name, research_data, system_prompt, user_prompt):
    log_message(f"Generating Wikipedia-style article for project: {project_name}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(project_name=project_name, research_data=research_data)},
    ]

    response = aiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return response.choices[0].message.content

# Async function to process each project
async def process_projects(projects_file_path, prompts_file_path):
    try:
        # Load projects and prompts
        projects_data = pd.read_excel(projects_file_path)
        prompts_data = load_prompts(prompts_file_path)

        if projects_data.empty:
            log_message("The projects file contains no data.")
            return

        # Add "Article" column if it doesn't exist
        if "Article" not in projects_data.columns:
            projects_data["Article"] = ""

        log_message("Processing projects and generating research and articles:")

        for index, row in tqdm(
            projects_data.iterrows(), total=projects_data.shape[0], unit="project", colour="yellow"
        ):
            project_name = row.get("Name", f"Project_{index+1}")

            # Handle empty or NaN categories and default to "projects"
            category = row.get("Category")
            if pd.isna(category) or str(category).strip() == "":
                category = "projects"
            else:
                category = str(category).strip()

            # Ensure prompts_data["Category"] is processed for string operations
            prompts_data["Category"] = prompts_data["Category"].astype(str).str.strip()

            # Fetch the relevant prompts for the category
            category_prompts = prompts_data[prompts_data["Category"].eq(category)]
            if category_prompts.empty:
                log_message(f"No prompts found for category: {category}")
                continue

            # Sort prompts by index
            category_prompts = category_prompts.sort_values("index")

            # Initialize previous output for chaining
            prev_output = ""

            # Process each prompt
            for _, prompt_row in category_prompts.iterrows():
                model = prompt_row["model"].strip().lower()
                system_prompt = prompt_row["System prompt"]
                user_prompt_template = prompt_row["Prompt"]

                # Replace placeholders in the user prompt
                user_prompt = user_prompt_template.replace("${project_name}", project_name).replace("${prev_output}", prev_output)

                # Sanitize project name for filenames
                sanitized_project_name = sanitize_filename(project_name)

                # Handle "researcher" and "gpt" models
                if model == "researcher":
                    result = await fetch_research_data(project_name, system_prompt, user_prompt)
                elif model == "gpt":
                    result = await generate_article_from_research(project_name, prev_output, system_prompt, user_prompt)
                else:
                    log_message(f"Unsupported model: {model}")
                    continue

                # Save the result to a file
                file_path = f"research/{sanitized_project_name}_{prompt_row['index']}.md"
                save_to_file(file_path, result)

                # Update prev_output for the next prompt
                prev_output = result

            # Save the final output of the last "gpt" prompt as the article
            
            # article_file_path = f"articles/{sanitized_project_name}.wiki"
            # save_to_file(article_file_path, prev_output)
            
            last_md_file = f"research/{sanitized_project_name}_{prompt_row['index']}.md"
            wiki_file_path = f"articles/{sanitized_project_name}.wiki"

            try:
                # Run pandoc to convert the last .md file to a .wiki file
                subprocess.run(
                    ["pandoc", last_md_file, "-t", "mediawiki", "-o", wiki_file_path],
                    check=True
                )
                print(f"Successfully generated {wiki_file_path} from {last_md_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error during pandoc conversion: {e}")
            except FileNotFoundError:
                print("Pandoc is not installed or not in the system PATH.")

            # Update the "Article" column in the XLSX file
            # projects_data.at[index, "Article"] = prev_output

        # Save the updated XLSX file
        # projects_data.to_excel(projects_file_path, index=False)
        log_message("Processing completed, articles generated successfully")

    except Exception as e:
        log_message(f"Error: {e}")

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Generate research and Wikipedia-style articles for projects in an XLSX file."
    )

    parser.add_argument(
        "--projects-file",
        type=str,
        default="resources/projects.xlsx",
        help='Path to the projects XLSX file. Default is "resources/projects.xlsx".',
    )

    parser.add_argument(
        "--prompts-file",
        type=str,
        default="resources/prompts.xlsx",
        help='Path to the prompts XLSX file. Default is "resources/prompts.xlsx".',
    )

    args = parser.parse_args()

    asyncio.run(process_projects(args.projects_file, args.prompts_file))

if __name__ == "__main__":
    main()
