
<p align="center">
<img src="./assets/glotlid_logo.svg" alt="GlotLID" width="30%" />
</p>
<p align="center">
<a href="https://huggingface.co/cis-lmu/glotlid"><img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-8A2BE2"></a>
<a href="https://huggingface.co/spaces/cis-lmu/glotlid-space"><img alt="HuggingFace Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space (Demo)-orange"></a>
<a href="https://github.com/cisnlp/GlotLID/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/cisnlp/GlotLID?logoColor=blue"></a>
<a href="."><img alt="GitHub stars" src="https://img.shields.io/github/stars/cisnlp/GlotLID"></a>
<a href="https://arxiv.org/abs/2310.16248"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.16248-b31b1b.svg"></a>
</p>

## TL;DR

The repository introduces **GlotLID**, an open-source language identification model with support for more than **1600 languages**.


## How to use

### Language Identification (Python)

You can use the model directly with fasttext library to predict language label:

```python
! pip install fasttext
! pip install huggingface_hub
```

```python
import fasttext
from huggingface_hub import hf_hub_download

# download model
## cache_dir: path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

# load the model
model = fasttext.load_model(model_path)

# predict language label (call this function as many times as needed)
model.predict("Hello, world!")
```

### Sentence Vectors (Python)

You can also use the model with fasttext library to get sentence vectors:

```python
! pip install fasttext
! pip install huggingface_hub
```

```python
import fasttext
from huggingface_hub import hf_hub_download

# download model
## cache_dir: path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

# load the model
model = fasttext.load_model(model_path)

# get sentence vector of input sentence (call this function as many times as needed)
embedding = model.get_sentence_vector(sent)
```

## Versions

We always maintain the previous version of GlotLID in our [huggingface](https://huggingface.co/cis-lmu/glotlid) repository.

To access a specific version, simply append the version number to the filename.

For v1: `model_v1.bin` (introduced in the GlotLID paper and used in all experiments).

For v2: `model_v2.bin` (an edited version of v1, featuring more languages, and cleaned from noisy corpora based on the analysis of v1).
- It suuports 1802 three-letter iso codes (1847 three letter iso codes with script)
- For 1626 three-letter iso codes; v2 on the test set achieved F1 of 0.996 and FPR of 0.0002.
  - These 1626 languages are selected based on the 0.5 F1 threshold and 0.0005 FPR threshold for low resource languages.

`model.bin` always refers to the latest version (v2 now).


## Data Sources 

See list of data sources [here](./sources.md).

You're welcome to open a [pull request](https://github.com/cisnlp/GlotLID/pulls) or ([issue](https://github.com/cisnlp/GlotLID/issues)) and contribute new resources to our data list. Even for the languages we already support, we're actively seeking additional resources to mitigate domain shift issues.


## Benchmark 

- UDHR: access our clean version of udhr [here](https://huggingface.co/datasets/cis-lmu/udhr-lid).
- FLORES-200: devtest part of [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

## Evaluation

You can find how we compute F1, Recall, Precision, and FPR in [metrics.py](./assets/metrics.py).


## FAQ
- If you see wrong predicted tags by GlotLID for a normal long text open an [issue](https://github.com/cisnlp/GlotLID/issues), however:
  - if the script is not supported by our model then use [GlotScript](https://github.com/cisnlp/GlotLID) to verify for the predicted `lang_script`, script in the sentence exists!  Otherwise, you need to write a function that returns 'und_mainscript' in this situations. GlotScript can identify both the mainscript and all available scripts in the sentence. We recommend using GlotLID in conjunction with GlotScript.
  - The high confidence threshold for each language could be different. This is because not all languages have the same distance from each other. For one language, 0.6 is a lot because it is very close to a similar language (such as dyu and bam), while for another, 0.9 might not be. 
  - This model is primarily trained on longer sentences, avoid using it on very short sentences. Other language identification models are not good at short sentences as well unless you increase the ngram size, which is computationally expensive.
  -  In GlotLID, the false positive rate (FPR) for high-resource languages is much higher than for low-resource languages. However, even with this higher FPR, it is still lower than in a situation where the language identification model only recognizes high-resource languages. We are also okay with this situation since our main concern is for the FPR of low-resource languages to be low. The high-resource base frequency is much higher than for low-resource languages, so cleanliness would not be a threat for those languages. However, for a low-resource language with a low base frequency, even a small FPR might result in most of the corpus being noisy.

- If you want to add a language, provide the resource in an open [issue](https://github.com/cisnlp/GlotLID/issues), and we will add it. If you require the model urgently, we can expedite the process in less than a week (the training itself takes less than a day). However, if there's no immediate urgency, that language will be included in the official release according to our schedule (depends on new resources).-
- If you need a custmoized model with susbet of languages let us known in an open [issue](https://github.com/cisnlp/GlotLID/issues)
- If you want to collaborate, please send us an email (to: amir@cis.lmu.de) specifying the type of collaboration you need from us.
- for the rest of requests feel free to email or open an issue.

## Media Coverage

See list of media coverage [here](./media.md).

## Citation

If you find our model, code and list of data sources useful for your research, please cite:

```
@inproceedings{
  kargaran2023glotlid,
  title={GlotLID: Language Identification for Low-Resource Languages},
  author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  url={https://openreview.net/forum?id=dl4e3EBz5j}
}
```


