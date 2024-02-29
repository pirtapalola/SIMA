
# Import libraries
import os

folder_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/' \
              'Methods_Ecolight/Jan2024_lognormal_priors/simulated_dataset/simulated_dataset'
file_runlist_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/' \
                    'Methods_Ecolight/Jan2024_lognormal_priors/runlist_all.txt'
output_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/' \
                   'Methods_Ecolight/Jan2024_lognormal_priors/runlist.txt'


# Specify the number of characters to remove from the start and end
start_slice = 14
end_slice = 22

# List to store sliced strings
sliced_strings = []

# Read the text file and slice each string
with open(file_runlist_path, 'r') as file:
    for line in file:
        # Remove specified number of characters from the start and end
        sliced_string = line[start_slice:-end_slice].strip()
        sliced_strings.append(sliced_string)

print(sliced_strings[0])


# Lists to store files
matching_files = []  # Files containing sliced strings
non_matching_files = []  # Files not containing sliced strings

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is a regular file
    if os.path.isfile(file_path):
        # Check if any sliced string is present in the filename
        if any(sliced_string in filename for sliced_string in sliced_strings):
            matching_files.append(file_path)
        else:
            non_matching_files.append(file_path)


# Print
print("Number of matching files: ", len(matching_files))
print("Number of non-matching files: ", len(non_matching_files))

# Read the content of the text file
with open(file_runlist_path, 'r') as text_file:
    lines = text_file.readlines()

# Identify lines to be removed
lines_to_remove = [line for line in lines for sliced_string in sliced_strings if sliced_string in line]
print("Lines to remove: ", lines_to_remove)

# Remove identified lines
updated_lines = [line for line in lines if line not in lines_to_remove]
print("Updated lines: ", updated_lines)

# Write the updated content to the new text file
with open(output_file_path, 'w') as new_text_file:
    new_text_file.writelines(updated_lines)
