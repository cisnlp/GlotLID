import os
import re
import subprocess
import pycountry

# ==============================
# CONFIG
# ==============================

REPO_URL = "https://github.com/google/fonts.git"
CLONE_DIR = "google_fonts_repo"
LANG_DIR = os.path.join(CLONE_DIR, "lang/Lib/gflanguages/data/languages")
OUTPUT_DIR = "googlefonts_output"

# ==============================
# ISO 639-1 → ISO 639-3
# ==============================

def get_iso_639_3(alpha2):
    try:
        lang = pycountry.languages.get(alpha_2=alpha2)
        if lang and hasattr(lang, "alpha_3"):
            return lang.alpha_3
    except Exception:
        pass
    return alpha2  # fallback if not found


# ==============================
# Extract + Split + Deduplicate
# ==============================

def extract_texts(content):
    blocks = []

    keys = ["styles", "tester", "poster_sm", "poster_md", "poster_lg"]

    for key in keys:
        matches = re.findall(rf'{key}:\s*"(.+?)"', content, re.DOTALL)
        blocks.extend(matches)

    specimens = re.findall(r'specimen_\d+:\s*"(.+?)"', content, re.DOTALL)
    blocks.extend(specimens)

    seen = set()
    final_lines = []

    for block in blocks:
        lines = block.split("\\n")

        for line in lines:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)

            if not line:
                continue

            if line not in seen:
                seen.add(line)
                final_lines.append(line)

    return final_lines


# ==============================
# Clone Repository (Improved)
# ==============================

if not os.path.isdir(CLONE_DIR):
    print("📥 Cloning Google Fonts repository (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
        check=True
    )
else:
    if os.path.isdir(os.path.join(CLONE_DIR, ".git")):
        print("ℹ️ Repository already exists. Skipping clone.")
    else:
        raise RuntimeError(
            f"Directory '{CLONE_DIR}' exists but is not a valid git repository."
        )

# Verify language directory exists
if not os.path.isdir(LANG_DIR):
    raise FileNotFoundError(f"Language directory not found: {LANG_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Process All Language Files
# ==============================

for filename in os.listdir(LANG_DIR):
    if not filename.endswith(".textproto"):
        continue

    full_path = os.path.join(LANG_DIR, filename)

    with open(full_path, encoding="utf-8") as f:
        content = f.read()

    id_match = re.search(r'id:\s*"([^"]+)"', content)
    if not id_match:
        continue

    lang_full = id_match.group(1)
    parts = lang_full.split("_")
    if len(parts) != 2:
        continue

    lang_code, script = parts
    lang_code = lang_code.lower()

    if len(lang_code) == 2:
        lang3 = get_iso_639_3(lang_code)
    else:
        lang3 = lang_code

    texts = extract_texts(content)
    if not texts:
        continue

    if script in [
        'Arab', 'Beng', 'Cyrl', 'Ethi',
        'Hani', 'Hung', 'Jpan', 'Khmr', 'Mlym',
        'Mtei', 'Nkoo', 'Orya', 'Taml', 'Tfng', 'Thai',
        'Armn', 'Cans', 'Copt', 'Deva', 'Geor', 'Gujr', 'Hang',
        'Hebr', 'Java', 'Kali', 'Knda', 'Latn', 'Lisu',
        'Mymr', 'Olck', 'Sinh', 'Thaa', 'Tibt', 'Hans', 'Hant', 'Hira', 'Kana', 'Grek', 'Kore', 'Telu', 'Syrc', 'Guru']:
        continue

    # Create script folder inside OUTPUT_DIR
    # script_dir = os.path.join(OUTPUT_DIR, script)
    # os.makedirs(script_dir, exist_ok=True)

    out_filename = f"{lang3}_{script}_googlefonts.txt"

    out_path = os.path.join(OUTPUT_DIR, out_filename)

    with open(out_path, "w", encoding="utf-8") as out:
        for line in texts:
            out.write(line + "\n")

    print(f"✔ Created {out_filename}")

print("\n🎉 Finished processing all languages.")