import re


def extract_value_from_line(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the numerical value using a regular expression
    pattern = r'(\d+\.\d+E[+-]\d+)'
    match = re.search(pattern, lines[line_number - 1])

    if match:
        return match.group(0)
    else:
        return None


def process_files(file_paths, line_number):
    results = []
    for file_path in file_paths:
        result = extract_value_from_line(file_path, line_number)
        results.append(result)
    return results


# Replace 'path/to/your/files' with the actual path to the directory containing your text files
directory_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Dec2023_lognormal_priors/' \
                 'EL_test_2_dec2023/EL_test_2_dec2023'

# List all text files in the specified directory
import os
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]

# Specify the line number to extract the value
line_number = 14

# Process files and save results in a list
results_list = process_files(file_paths, line_number)

# Print the results
for i, result in enumerate(results_list):
    print(f"File {i + 1}: {'No value found' if result is None else result}")
