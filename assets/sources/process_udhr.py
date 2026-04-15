import requests
import xml.etree.ElementTree as ET
import re

# 1) Download the XML file; for example eng_Shaw here
url = "https://raw.githubusercontent.com/twardoch/udhr-custom/refs/heads/main/data/udhr-manual/udhr_eng_shaw.xml"
response = requests.get(url)
response.raise_for_status()

xml_data = response.text

# 2) Parse XML
root = ET.fromstring(xml_data)

# 3) Recursively extract all text
def extract_text(element):
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.extend(extract_text(child))
        if child.tail:
            texts.append(child.tail)
    return texts

raw_text = extract_text(root)

# 4) Clean text
cleaned = []
seen = set()

for line in raw_text:
    # Strip whitespace
    line = line.strip()

    # Remove numbers
    line = re.sub(r'\d+', '', line)

    # Remove extra spaces
    line = re.sub(r'\s+', ' ', line)

    # Remove empty lines
    if not line:
        continue

    # Remove duplicates
    if line not in seen:
        seen.add(line)
        cleaned.append(line)

# 6) Save cleaned file
with open("eng_Shaw_udhr.txt", "w", encoding="utf-8") as f:
    for line in cleaned:
        f.write(line + "\n")