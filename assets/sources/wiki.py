# [step 1] create a dict, the keys are wikipedia codes, the values are the iso-639-3 codes, You can change them as you fit, add more langs or delete
wiki_low = {'cbk-zam': 'cbk', 'anp': 'anp',
'trv': 'trv', 'frp': 'frp', 'arc': 'syc',
'tly': 'tly', 'tay': 'tay',
'szy': 'szy', 'pwn': 'pwn', 'pi': 'pli',
'pih': 'pih', 'nrm': 'nrm', 'nqo': 'nqo',
'nov': 'nov', 'inh': 'inh',
'be-tarask': 'bel', 'atj': 'atj',
'blk': 'blk', 'cdo': 'cdo', 'chy': 'chy', 'dag': 'dag', 'gan': 'gan'}

# [step 2] create wiki_low folder
! mkdir ./wiki_low
! mkdir ./wiki_low/csv

# [step 3] download parquet files
import requests
from datasets import load_dataset

for k in tqdm(wiki_low.keys()):
    try:
        url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.{k}/train-00000-of-00001.parquet?download=true"
        filename = f"./wiki_low/{wiki_low[k]}.parquet"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the file in binary write mode and write the content
            with open(filename, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully as", filename)
        else:
            print("Failed to download file, status code:", response.status_code)
    except:
        pass


# [step 4] read parquet files, process them and save them 
from datasets import load_dataset
from GlotScript import sp
from tqdm import tqdm 
import pandas as pd

for k in wiki_low.keys():
    try:
        filename = f"./wiki_low/{wiki_low[k]}.parquet"
        df = pd.read_parquet(filename)
        df['script'] = df['text'].apply(lambda x: sp(x)[0])
        df['script_value'] = df['text'].apply(lambda x: sp(x)[1])

        main_script = df['script'].value_counts().idxmax()
        df = df[df['script'] == main_script]
        df = df[df['script_value'] > 0.98] 
        list_of_lists = [l.split('\n') for l in df['text']]

        texts = [item for sublist in list_of_lists for item in sublist]
        
        if main_script == 'Latn': 
            texts = [t.strip() for t in texts if len(t) > 100]
        else:
            texts = [t.strip() for t in texts if len(t) > 50]

        file_path = f"./wiki_low/csv/{wiki_low[k]}_{main_script}_HFWikipedia.txt"

        # Open the file in write mode
        with open(file_path, "w") as file:
            # Write each item from the list to the file
            for text in texts:
                file.write(text + "\n")  # Add a newline after each item
    except:
        pass
