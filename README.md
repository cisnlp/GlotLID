
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

You can use the model directly with fasttext library:

```python
! pip install fasttext
! pip install huggingface_hub
```

```python
import fasttext
from huggingface_hub import hf_hub_download

# Download model
## cache_dir: Path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

# Load the model
model = fasttext.load_model(model_path)

# Predict using the model (call this function as many times as needed)
model.predict("Hello, world!")
```

## Data Sources

See list of data sources [here](./sources.md).

## Benchmark 

- UDHR: access our clean version of udhr [here](https://huggingface.co/datasets/cis-lmu/udhr-lid).
- FLORES-200: devtest part of [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

## Evaluation

Codes will be uploaded soon.

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


