import pandas as pd
import argparse
from tqdm import tqdm
import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

API_KEY = os.getenv("PERPLEXITY_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

# Create research directory if not exists
os.makedirs("research", exist_ok=True)

# async function to fetch research data from Perplexity API
async def fetch_research_data(project_name, website):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with gathering detailed research "
                "information on projects based on a provided name and URL."
            ),
        },
        {
            "role": "user",
            "content": f"Provide research information about the project '{project_name}'.",
        },
    ]
    
    # Retrieve the completion response in one go (non-streaming mode here for simplicity)
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    
    # Extract the main response content and citations
    content =  response.choices[0].message.content
    citations = response.citations

    # Format the output
    formatted_output = f"### Research Data on '{project_name}'\n\n"
    formatted_output += content
    if citations:
        formatted_output += "\n\n### Citations\n"
        formatted_output += "\n".join(f"{i+1}. {citation}" for i, citation in enumerate(citations))
    
    return formatted_output

    

# function to save research info to a file
def save_research_info(project_name, research_info):
    # Define file path
    file_path = f"research/{project_name}.txt"
    with open(file_path, "w") as file:
        file.write(research_info)
    print(f"Saved research info for '{project_name}' to {file_path}")

# async function to process each project
async def generate_research(projects_file_path):
    try:
        data = pd.read_excel(projects_file_path)
        
        # Check if there are any rows in the file
        if data.empty:
            print("The file contains no data.")
            return
        
        # Iterate through each row and generate research info
        print("Generating research information for each project:")
        
        for index, row in tqdm(data.iterrows(), total=data.shape[0], unit="project", colour="blue"):
            # Extract project info
            project_name = row.get("Name", "Unknown Project")
            description = row.get("Description", "No description available.")
            website = row.get("Website", "No website provided.")
            
            # Fetch research data
            additional_research_data = await fetch_research_data(project_name, website)
            
            # Compile research info
            research_info = (
                f"Project: {project_name}\n"
                f"Description: {description}\n"
                f"Website: {website}\n"
                f"Additional Research: {additional_research_data}\n"
                f"Status: In Research Phase\n"
            )

            # Save research info to a file
            save_research_info(project_name, research_info)

        print("\nResearch information generation completed.")

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate research information for projects in an XLSX file.")
    
    # Add optional argument for the file path
    parser.add_argument('--file-path', type=str, default='resources/projects.xlsx', 
                        help='Path to the XLSX file. Default is "resources/projects.xlsx".')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the async generate_research function
    asyncio.run(generate_research(args.file_path))

if __name__ == "__main__":
    main()
