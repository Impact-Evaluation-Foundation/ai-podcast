import json
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
perplexity_client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai/")
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
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(content)
    print(f"Saved content to {file_path}")

# Utility: Sanitize filenames by removing special characters
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?* ]', "_", name)

# Load prompts from prompts.xlsx
def load_prompts(prompts_file_path):
    return pd.read_excel(prompts_file_path)


def convert_json_to_infobox(json_data):
    # Start the infobox
    infobox = "{{Infobox(Generic Article)\n"

    # Adding each field from JSON to the infobox
    infobox += f"| title = {json_data.get('title', 'Unknown Title')}\n"
    infobox += f"| overview = {json_data.get('overview', 'No overview provided.')}\n"
    infobox += f"| importance = {json_data.get('importance', 'Unknown')}\n"

    # Handling related topics (comma-separated)
    related_topics = ", ".join(json_data.get('related_topics', []))
    infobox += f"| related_topics = {related_topics}\n"

    # Handling external resources (formatted as links)
    external_resources = ""
    for resource in json_data.get('external_resources', []):
        url = resource.get('url', '')
        description = resource.get('description', '')
        if url and description:
            external_resources += f"[{url} {description}], "
    
    # Remove the trailing comma and space, if any
    if external_resources.endswith(", "):
        external_resources = external_resources[:-2]

    infobox += f"| external_resources = \n{external_resources}\n"

    # Closing the infobox
    infobox += "}}"

    return infobox

# Async function to fetch research data with Tavily x GPT
async def fetch_research_data(project_name, system_prompt, user_prompt):
    log_message(f"Fetching research data for project: {project_name}")
    researcher = GPTResearcher(user_prompt, "research_report")
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    
    references_index = report.find("## References")
    content_below_references = ""
    if references_index != -1:
        # Skip "## References" heading and any trailing whitespace/newlines
        content_below_references = report[references_index + len("## References"):].lstrip()
    
    return content_below_references

# Utility: Generate the Infobox JSON
async def generate_infobox_json(project_name, prev_output, infobox_system_prompt, infobox_user_prompt):
    # Replace placeholders in the infobox user prompt
    user_prompt = infobox_user_prompt.replace("${project_name}", project_name).replace("${prev_output}", prev_output)
    
    # Create the system and user messages for the prompt
    messages = [
        {"role": "system", "content": infobox_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generate the response using aiClient
    response = aiClient.chat.completions.create(
        model="gpt-4o-mini",  # or the appropriate model
        messages=messages
    )
    
    # Parse the response to get the JSON object
    infobox_json = response.choices[0].message.content.strip()
    
    # Assuming the response is valid JSON, load it into a dictionary
    try:
        infobox_data = json.loads(infobox_json)
    except json.JSONDecodeError:
        print("Error parsing the JSON response from the AI model.")
        return ""

    # Convert the JSON to the MediaWiki infobox format
    return convert_json_to_infobox(infobox_data)

# Async function to generate content using a specific system prompt
async def generate_article_from_research(project_name, research_data, system_prompt, user_prompt):
    log_message(f"Generating content for project w GPT-4o: {project_name}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(project_name=project_name, research_data=research_data)},
    ]

    response = aiClient.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    return response.choices[0].message.content

# Generate content using Perplexity AI
async def generate_perplexity_research(project_name, research_data, system_prompt, user_prompt):
    log_message(f"Generating content using Perplexity for project: {project_name}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(project_name=project_name, research_data=research_data)},
    ]

    response = perplexity_client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    return response.choices[0].message.content

# Async function to process each project
async def process_projects(projects_file_path, prompts_file_path, infobox_prompts_file_path):
    try:
        # Load projects, prompts, and infobox prompts
        projects_data = pd.read_excel(projects_file_path)
        prompts_data = load_prompts(prompts_file_path)
        infobox_prompts_data = pd.read_excel(infobox_prompts_file_path)

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

            # Process each prompt in the category (research or article generation)
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

            # Now, fetch and generate the Infobox JSON
            # Extract the first infobox prompt for the selected category (assuming only one prompt is used)
            infobox_prompt_row = infobox_prompts_data[infobox_prompts_data["Category"].eq(category)].iloc[0]
            infobox_system_prompt = infobox_prompt_row["System prompt"]
            infobox_user_prompt = infobox_prompt_row["Prompt"]

            infobox_json = await generate_infobox_json(project_name, prev_output, infobox_system_prompt, infobox_user_prompt)

            # Prepend the infobox to the article
            last_md_file = f"research/{sanitized_project_name}_{prompt_row['index']}.md"
            wiki_file_path = f"articles/{sanitized_project_name}.wiki"

            try:
                # Ensure the .md file is properly encoded in UTF-8 before passing it to pandoc
                with open(last_md_file, 'r', encoding='utf-8', errors='replace') as md_file:
                    md_content = md_file.read()

                # Run pandoc to convert the last .md file to a .wiki file, specifying utf-8 encoding
                subprocess.run(
                    ["pandoc", last_md_file, "-t", "mediawiki", "-o", wiki_file_path],
                    check=True
                )
                print(f"Successfully generated {wiki_file_path} from {last_md_file}")
    
                # After pandoc success, prepend the infobox to the .wiki file
                with open(wiki_file_path, "r+", encoding='utf-8') as file:
                    content = file.read()
                    # Prepend the infobox JSON with a newline below it
                    file.seek(0, 0)
                    file.write(f"{infobox_json}\n\n" + content)

            except subprocess.CalledProcessError as e:
                print(f"Error during pandoc conversion: {e}")
            except FileNotFoundError:
                print("Pandoc is not installed or not in the system PATH.")
            except Exception as e:
                print(f"Error: {e}")


            # Update the "Article" column in the XLSX file
            # projects_data.at[index, "Article"] = prev_output

        # Save the updated XLSX file
        # projects_data.to_excel(projects_file_path, index=False)
        log_message("Processing completed, articles generated successfully")

    except Exception as e:
        log_message(f"Error: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process project data and generate articles with infoboxes.")
    parser.add_argument(
        '--infobox-prompts-file', 
        type=str, 
        default='resources/infobox_prompts.xlsx', 
        help="Path to the custom infobox prompts file (default: resources/infobox_prompts.xlsx)"
    )
    parser.add_argument(
        '--projects-file', 
        type=str, 
        default="resources/projects.xlsx", 
        help="Path to the projects file"
    )
    parser.add_argument(
        '--prompts-file', 
        type=str, 
        default="resources/prompts.xlsx",
        help="Path to the prompts file"
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Call the process_projects function with the custom infobox prompts file
    import asyncio
    asyncio.run(process_projects(args.projects_file, args.prompts_file, args.infobox_prompts_file))
