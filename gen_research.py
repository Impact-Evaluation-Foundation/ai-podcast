import pandas as pd
import argparse
from tqdm import tqdm
import asyncio

# async function to fetch research data
async def fetch_research_data(project_name):
    await asyncio.sleep(0.5)  # Simulates the delay of an API call
    # Mocked research data (in a real scenario, this would be data from an API)
    return f"Research data for {project_name}"

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
            additional_research_data = await fetch_research_data(project_name)
            
            # Generate boilerplate research info
            research_info = (
                f"Project: {project_name}\n"
                f"Description: {description}\n"
                f"Website: {website}\n"
                f"Additional Research: {additional_research_data}\n"
                f"Status: In Research Phase\n"
            )

            # Print research info for each project
            print("\n" + "="*40)
            print(research_info)
            print("="*40)

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
