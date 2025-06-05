main_source = "/path/to/folder_of_lang_script_source/"

# /path/to/folder_of_lang_script_source/
# ├── deu_Latn_wikipedia.txt
# ├── deu_Latn_bible.txt
# ├── eng_Latn_wikipedia.txt


import os
import pandas as pd

directory_path = main_source

file_names = []
script_names = []
source_names = []
iso_names = []
file_sizes = []

files = os.listdir(directory_path)

for file in files:
    file_path = os.path.join(directory_path, file)
    if os.path.isfile(file_path) and file.endswith(".txt"):
        parts = file.split('_')
        if len(parts) == 3:
            iso, script, source = parts
            size = os.path.getsize(file_path)
            source = source.replace('.txt', '')
            file_names.append(file)
            script_names.append(script)
            source_names.append(source)
            iso_names.append(iso)
            file_sizes.append(size)

data = {
    'file_name': file_names,
    'iso': iso_names,
    'script': script_names,
    'source': source_names,
    'size(bytes)': file_sizes
}

df_all_new = pd.DataFrame(data)



script_counts = df_all_new.groupby('iso')['script'].nunique().reset_index()

# Rename the column to 'script_count' for clarity
script_counts.rename(columns={'script': 'script_count'}, inplace=True)

# Display the resulting DataFrame
script_counts = script_counts.sort_values('script_count', ascending=False)


df_all_new_cutoff_size = df_all_new[df_all_new['size(bytes)'] > 1_000_000]
script_counts = df_all_new_cutoff_size.groupby('iso')['script'].nunique().reset_index()

# Rename the column to 'script_count' for clarity
script_counts.rename(columns={'script': 'script_count'}, inplace=True)

# Display the resulting DataFrame
len(script_counts.sort_values('script_count', ascending=False))


df_iso_script_pair = df_all_new.groupby(['iso', 'script'])['size(bytes)'].sum().reset_index()
df_iso_script_pair = df_iso_script_pair.sort_values('size(bytes)', ascending=False)
df_iso_script_pair_restrict = df_iso_script_pair[df_iso_script_pair['size(bytes)']> 0]
# df_iso_script_pair_restrict = df_iso_script_pair[df_iso_script_pair['size(bytes)']> 5120 * 1]

len(set(df_iso_script_pair_restrict['iso'])), len(df_iso_script_pair_restrict)


# ! wget https://raw.githubusercontent.com/cisnlp/GlotScript/main/metadata/GlotScript.tsv
df_meta = pd.read_csv('GlotScript.tsv', na_filter= False, sep='\t')


import pandas as pd

# Assuming 'iso' is the column in df_iso_script_pair_restrict, and 'ISO639-3' is the column in df_meta
result_df = pd.merge(df_iso_script_pair_restrict, df_meta, how='left', left_on='iso', right_on='ISO639-3')


import pandas as pd

# Assuming df_iso_script_pair_restrict is your DataFrame
# and 'script' and 'ISO15924-Main' are the respective column names

# Create an empty list to store the rows that meet the condition
result_rows = []

# Iterate over the rows of the DataFrame
for index, row in result_df.iterrows():
    if row['script'] not in str(row['ISO15924-Main']):
        if row['iso']!= 'zxx':
            result_rows.append(row)

# Create a new DataFrame from the result_rows list
errors_script = pd.DataFrame(result_rows)

errors_script[errors_script['iso']!='und']


df_iso_script_pair_restrict['iso_script'] = df_iso_script_pair_restrict.apply(lambda row: row[
    'iso'] + '_' + row['script'], axis=1)


df_iso_script_pair_restrict


import os
import pandas as pd
from tqdm import tqdm

# Directory containing the .txt files
directory = main_source

# Output directory for merged files
output_directory = "/path/to/merge_source"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Create a dictionary to store merged content for each pair
merged_content = {}

# Loop through files in the directory
for filename in tqdm(os.listdir(directory)):
    filepath = os.path.join(directory, filename)

    # Check if it's a file and ends with .txt
    if filename.endswith(".txt"):
        # Extract ISO and script pair from the filename
        parts = filename.split("_", 2)
        if len(parts) == 3:  # Assuming the format is "iso_script_source.txt"
            iso = parts[0]
            script = parts[1]
            pair = f"{iso}_{script}"

            # Check if the pair is in the DataFrame
            if pair in df_iso_script_pair_restrict['iso_script'].values:
                # Read the content of the file
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()

                # Determine the output file path
                merged_filepath = os.path.join(output_directory, f"{pair}.txt")

                # Write the content to the output file with "w" or "a" method
                with open(merged_filepath, "a" if pair in merged_content else "w", encoding="utf-8") as merged_file:
                    merged_file.write(content)

                # Mark the pair as processed
                merged_content[pair] = True

print("Files have been merged and saved.")



import os
import random
import shutil
from tqdm import tqdm

# Function to shuffle and split sentences within a file
def shuffle_and_split_sentences(file_path, train_file, val_file, test_file, train_percent, val_percent, test_percent):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Deduplicate lines
    lines = list(set(lines))
    
    # Shuffle the lines within the file
    random.shuffle(lines)
    
    total_lines = len(lines)
    num_train = int(total_lines * train_percent)
    num_val = int(total_lines * val_percent)
    num_test = int(total_lines * test_percent)
    
    train_lines = lines[:num_train]
    val_lines = lines[num_train:num_train + num_val]
    test_lines = lines[num_train + num_val:]
    
    with open(train_file, 'a', encoding='utf-8') as train:
        train.writelines(train_lines)
    with open(val_file, 'a', encoding='utf-8') as val:
        val.writelines(val_lines)
    with open(test_file, 'a', encoding='utf-8') as test:
        test.writelines(test_lines)

# Path to the directory containing the merged files
merge_source_directory = "/path/to/merge_source"

# Output directories for train, validation, and test sets
output_directory = "/path/to/split_normal"
train_directory = os.path.join(output_directory, "train")
val_directory = os.path.join(output_directory, "val")
test_directory = os.path.join(output_directory, "test")

# Create output directories if they don't exist
os.makedirs(train_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# List of merged files in merge_source_directory
merged_files = os.listdir(merge_source_directory)

# Shuffle the list of files randomly
random.shuffle(merged_files)

# Define the percentages
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# Iterate through each file and split into train, validation, and test sets
for file in tqdm(merged_files):
    source_path = os.path.join(merge_source_directory, file)
    
    # Determine the target file paths
    train_file = os.path.join(train_directory, file)
    val_file = os.path.join(val_directory, file)
    test_file = os.path.join(test_directory, file)
    
    # Shuffle and split the sentences within the file
    shuffle_and_split_sentences(source_path, train_file, val_file, test_file, train_percent, val_percent, test_percent)

print("Files have been deduplicated, shuffled, and split into train, validation, and test sets based on sentence counts.")



import os
from tqdm import tqdm

# Directory containing the iso_script.txt files
train_directory = "/path/to/split_normal/train"

# Output directory for the combined train.txt file
output_directory = "/path/to/split_normal"

# Name of the combined train.txt file
output_file = "train.txt"

# Define the batch size
batch_size = 2000
lines_buffer = []  # Buffer to hold lines before writing

# Open the combined train.txt file for writing
with open(os.path.join(output_directory, output_file), "w", encoding="utf-8") as combined_file:
    # Iterate through the iso_script.txt files in the train directory
    for filename in tqdm(os.listdir(train_directory)):
        if filename.endswith(".txt"):
            iso_script = os.path.splitext(filename)[0]  # Extract iso_script label
            with open(os.path.join(train_directory, filename), "r", encoding="utf-8") as train_file:
                lines = train_file.readlines()
                # Write each line to the buffer with the label
                for line in lines:
                    lines_buffer.append(f"__label__{iso_script} {line}")
                    # If the buffer reaches the batch size, write it to the file
                    if len(lines_buffer) >= batch_size:
                        combined_file.writelines(lines_buffer)
                        lines_buffer = []  # Clear the buffer

    # Write any remaining lines in the buffer to the file
    combined_file.writelines(lines_buffer)

print("Combined train.txt file has been created in batches.")




import os
from tqdm import tqdm

# Directory containing the iso_script.txt files
train_directory = "/path/to/split_normal/val"

# Output directory for the combined train.txt file
output_directory = "/path/to/split_normal"

# Name of the combined train.txt file
output_file = "val.txt"

# Define the batch size
batch_size = 2000
lines_buffer = []  # Buffer to hold lines before writing

# Open the combined train.txt file for writing
with open(os.path.join(output_directory, output_file), "w", encoding="utf-8") as combined_file:
    # Iterate through the iso_script.txt files in the train directory
    for filename in tqdm(os.listdir(train_directory)):
        if filename.endswith(".txt"):
            iso_script = os.path.splitext(filename)[0]  # Extract iso_script label
            with open(os.path.join(train_directory, filename), "r", encoding="utf-8") as train_file:
                lines = train_file.readlines()
                # Write each line to the buffer with the label
                for line in lines:
                    lines_buffer.append(f"__label__{iso_script} {line}")
                    # If the buffer reaches the batch size, write it to the file
                    if len(lines_buffer) >= batch_size:
                        combined_file.writelines(lines_buffer)
                        lines_buffer = []  # Clear the buffer

    # Write any remaining lines in the buffer to the file
    combined_file.writelines(lines_buffer)

print("Combined val.txt file has been created in batches.")
