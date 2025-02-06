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

from utils import check_urls, handle_question_generator


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

async def generate_clean_content(project_name: str, bad_urls: list, original_content: str, system_prompt: str, user_prompt: str) -> str:
    """Generate cleaned content by removing/replacing invalid URLs"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(
            bad_urls="\n".join(bad_urls),
            original_content=original_content
        )}
    ]
    
    response = aiClient.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2  # Keep it factual
    )
    
    return response.choices[0].message.content.strip()

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
            context = {
                # Project metadata
                "project_name": row.get("Name", f"Project_{index+1}"),
                "category": str(row.get("Category", "projects")).strip(),
                "sanitized_name": sanitize_filename(row.get("Name", f"Project_{index+1}")),
                "topic":row.get("Name", f"Project_{index+1}"),
                # Research components
                "prev_output": "",
                "question_1": None,
                "question_2": None,
                "question_3": None,
                "question_4": None,
                "question_5": None,
                "question_6": None,
                "question_7": None,
                "question_8": None,
                "question_9": None,
                "question_10": None,
                "tavily_question": None,
                "tavily_question_answers_references": None,
                "list_of_perplexity_questions_answers_references": [],
                
                # System tracking
                "research_files": [],
                "errors": []
            }

            # Sanitize category
            if pd.isna(context["category"]) or context["category"] == "":
                context["category"] = "projects"

            try:
                # Load relevant prompts
                prompts_data["Category"] = prompts_data["Category"].astype(str).str.strip()
                category_prompts = prompts_data[prompts_data["Category"].eq(context["category"])]
                if category_prompts.empty:
                    log_message(f"No prompts found for category: {context['category']}")
                    continue

                # Process prompts in index order
                for _, prompt_row in category_prompts.sort_values("index").iterrows():
                    try:
                        # Get prompt configuration
                        model = prompt_row["model"].strip().lower()
                        prompt_type = prompt_row.get("Prompt type", "").strip().lower() or None
                        system_prompt = prompt_row["System prompt"]
                        user_template = prompt_row["Prompt"]
                        prompt_index = prompt_row["index"]

                        # Format user prompt with context
                        user_prompt = user_template
                        for key in context:
                            value = context[key]
                            if isinstance(value, list):
                                formatted_value = "\n\n".join(value)
                            else:
                                formatted_value = str(value or "")
                            user_prompt = user_prompt.replace(f"${{{key}}}", formatted_value)
                        
                        # Save the formatted prompt for debugging
                        prompt_debug_path = f"prompts/{context['sanitized_name']}_{prompt_index}.md"
                        save_to_file(prompt_debug_path, user_prompt)
                        
                        # Execute prompt based on type and model
                        result = None
                        if model == "researcher":
                            result = await fetch_research_data(
                                context["project_name"], 
                                system_prompt, 
                                user_prompt
                            )
                            if prompt_type == "tavily_answer":
                                context["tavily_question_answers_references"] = result

                        elif model == "gpt":
                            if prompt_type == "url_cleaner":
                                # Get references from context
                                tavily_refs = context.get('tavily_question_answers_references', '')
                                perplexity_refs = "\n".join(context.get('list_of_perplexity_questions_answers_references', []))
                
                                # Validate URLs and clean content
                                bad_urls = await check_urls(tavily_refs, perplexity_refs)
                                result = await generate_clean_content(
                                    context["project_name"],
                                    bad_urls,
                                    context["prev_output"],  # Content to clean
                                    system_prompt,
                                    user_prompt.replace("${bad_urls}", "\n".join(bad_urls))
                                )
                            else:
                                # Original article generation
                                result = await generate_article_from_research(
                                context["project_name"],
                                context["prev_output"],
                                system_prompt,
                                user_prompt
                                )
                            if prompt_type == "question_generator":
                                parsed = handle_question_generator(result)
                                context.update(parsed)
                           

                        elif model == "perplexity":
                            result = await generate_perplexity_research(
                                context["project_name"],
                                context["prev_output"],
                                system_prompt,
                                user_prompt
                            )
                            if prompt_type == "perplexity_answers":
                                context["list_of_perplexity_questions_answers_references"].append(result)

                        else:
                            log_message(f"Unsupported model: {model}")
                            continue

                        # Update context and save results
                        if result:
                            # Save output file
                            filename = f"{context['sanitized_name']}_{prompt_index}.md"
                            file_path = f"research/{filename}"
                            save_to_file(file_path, result)
                            context["research_files"].append(file_path)
                            
                            # Update previous output for legacy prompts
                            if not prompt_type:
                                context["prev_output"] = result
                            else:
                                context["prev_output"] = result  # Still update for chaining

                    except Exception as e:
                        error_msg = f"Prompt {prompt_index} failed: {str(e)}"
                        context["errors"].append(error_msg)
                        log_message(error_msg)
                        continue

                # Generate final infobox
                try:
                    infobox_prompt_row = infobox_prompts_data[
                        infobox_prompts_data["Category"].eq(context["category"])
                    ].iloc[0]

                    # Format infobox prompt with full context
                    infobox_user_prompt = infobox_prompt_row["Prompt"]
                    for key in context:
                        value = context[key]
                        infobox_user_prompt = infobox_user_prompt.replace(
                            f"${{{key}}}", 
                            "\n".join(value) if isinstance(value, list) else str(value or "")
                        )

                    infobox_json = await generate_infobox_json(
                        context["project_name"],
                        context["prev_output"],
                        infobox_prompt_row["System prompt"],
                        infobox_user_prompt
                    )

                    # Convert to wiki format
                    if context["research_files"]:
                        last_md = context["research_files"][-1]
                        wiki_path = f"articles/{context['sanitized_name']}.wiki"
                        
                        subprocess.run(
                            ["pandoc", last_md, "-t", "mediawiki", "-o", wiki_path],
                            check=True
                        )
                        
                        # Prepend infobox
                        with open(wiki_path, "r+", encoding='utf-8') as f:
                            content = f.read()
                            f.seek(0, 0)
                            f.write(f"{infobox_json}\n\n{content}")

                except Exception as e:
                    error_msg = f"Infobox generation failed: {str(e)}"
                    context["errors"].append(error_msg)
                    log_message(error_msg)

                # Update projects data
                projects_data.at[index, "Article"] = context["prev_output"]

            except Exception as e:
                error_msg = f"Project {context['project_name']} failed: {str(e)}"
                context["errors"].append(error_msg)
                log_message(error_msg)

        # Save updated projects data
        projects_data.to_excel(projects_file_path, index=False)
        log_message("Processing completed. Check 'articles' directory for outputs.")

    except Exception as e:
        log_message(f"Critical system error: {str(e)}")

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
