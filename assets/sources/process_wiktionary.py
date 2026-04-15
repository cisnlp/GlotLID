import os
from GlotScript import sp
from tqdm import tqdm

fonts_path = "../fonts"
wordlists_path = "/path/to/wiktionary_extract/wordlists" #https://github.com/iwsfutcmd/wiktionary_extract

# Output directory in current working directory
output_dir = os.path.join(os.getcwd(), "by_script")
os.makedirs(output_dir, exist_ok=True)

# Collect writing systems
writing_systems = {
    name for name in os.listdir(fonts_path)
    if os.path.isdir(os.path.join(fonts_path, name))
}

input_files = [
    f for f in os.listdir(wordlists_path)
    if os.path.isfile(os.path.join(wordlists_path, f))
]

for filename in tqdm(input_files, desc="Processing files"):
    file_path = os.path.join(wordlists_path, filename)
    output_files = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"{filename}", leave=False):
            line = line.strip()
            if not line:
                continue

            try:
                script = sp(line)[0]

                if sp(line)[1] < 0.85:
                    continue
            except Exception:
                continue

            if script in writing_systems:
                if script not in output_files:
                    output_filename = f"{filename}_{script}_wiktionary.txt"
                    output_path = os.path.join(output_dir, output_filename)
                    output_files[script] = open(output_path, "a", encoding="utf-8")

                output_files[script].write(line + "\n")

    for f in output_files.values():
        f.close()