import pandas as pd
import argparse
import os

def csv_to_xlsx(csv_file_path, xlsx_file_path):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Save it as an XLSX file
        data.to_excel(xlsx_file_path, index=False)
        
        print(f"Successfully converted {csv_file_path} to {xlsx_file_path}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert a CSV file to XLSX format.")
    
    # Add optional argument for the file path
    parser.add_argument('--file-path', type=str, default='resources/projects.csv', 
                        help='Path to the CSV file. Default is "resources/projects.csv".')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Define output XLSX file path by changing the extension to .xlsx
    file_path = args.file_path
    if file_path.endswith('.csv'):
        xlsx_file_path = file_path.replace('.csv', '.xlsx')
    else:
        xlsx_file_path = file_path + '.xlsx'

    # Convert the CSV to XLSX
    csv_to_xlsx(file_path, xlsx_file_path)

if __name__ == "__main__":
    main()
