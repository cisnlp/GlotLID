from tqdm import tqdm
import pandas as pd
import random
import os

def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)


directory_path = "/path/to/split_normal/train"

file_names = []
script_names = []
iso_names = []
file_sizes = []
lengths = []

files = os.listdir(directory_path)

for file in tqdm(files):
    file_path = os.path.join(directory_path, file)
    if os.path.isfile(file_path) and file.endswith(".txt"):
        parts = file.split('_')
        if len(parts) == 2:
            iso, script = parts
            size = os.path.getsize(file_path)
            script = script.replace('.txt', '')
            file_names.append(file)
            script_names.append(script)
            iso_names.append(iso)
            file_sizes.append(size)
            lengths.append(count_lines(file_path))

data = {
    'file_name': file_names,
    'iso': iso_names,
    'script': script_names,
    'len': lengths,
    'size(bytes)': file_sizes
}

df_all_new = pd.DataFrame(data)
print(df_all_new.sort_values('len', ascending=False))
total = sum(df_all_new['len'])
print(total)


def dist(row): 
    return ((row['len'] / total) ** 0.3)

df_all_new['dist'] = df_all_new.apply(dist, axis=1)


total_dist = sum(df_all_new['dist'])
print(total_dist)


df_all_new['sample_len'] = df_all_new['dist'].apply(lambda x: int((x /total_dist) * total ))
print(df_all_new.sort_values('len', ascending=False))
print(df_all_new[df_all_new['len'] < df_all_new['sample_len']].sort_values('len', ascending=False))



# Define paths
source_folder = '/path/to/split_normal/train'  # Change this to your source folder path
output_folder = '/path/to/split_normal/new_train/'  # Change this to your output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to perform sampling
def sample_sentences(input_file, sample_len):
    with open(input_file, 'r') as f:
        sentences = f.readlines()
    sampled_sentences = random.choices(sentences, k=sample_len)
    return sampled_sentences

# Iterate through the DataFrame and perform sampling
for _, row in df_all_new.iterrows():
    file_name = row['file_name']
    sample_len = row['sample_len']
    input_file = os.path.join(source_folder, file_name)

    # Check if the file exists
    if os.path.exists(input_file):
        if sample_len > row['len']:
            # Up-sample
            sampled_sentences = sample_sentences(input_file, sample_len)
        else:
            # Down-sample
            with open(input_file, 'r') as f:
                sampled_sentences = f.readlines()
        
        # Create a new file with sampled sentences
        output_file = os.path.join(output_folder, file_name)
        with open(output_file, 'w') as f:
            f.writelines(sampled_sentences)

# Confirm that the files have been created in the output folder
print(f"Files written to: {output_folder}")



# Directory containing the iso_script.txt files
train_directory = "/path/to/split_normal/new_train"

# Output directory for the combined train.txt file
output_directory = "/path/to/split_normal"

# Name of the combined train.txt file
output_file = "new_train.txt"

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

print(f"Combined {output_file} file has been created in batches.")



input_file = '/path/to/split_normal/new_train.txt'
output_file = '/path/to/split_normal/shuffled_new_train.txt'

# Read the lines from the input file
with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines to the output file
with open(output_file, 'w', encoding='utf-8') as file:
    file.writelines(lines)



