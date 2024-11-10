import pandas as pd

def csv_to_xlsx(csv_file_path, xlsx_file_path):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Save it as an XLSX file
        data.to_excel(xlsx_file_path, index=False)
        
        print(f"Successfully converted {csv_file_path} to {xlsx_file_path}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
csv_file_path = 'resources/projects.csv'
xlsx_file_path = 'resources/projects.xlsx'
csv_to_xlsx(csv_file_path, xlsx_file_path)
