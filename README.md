
<p align="center">
<img src="./assets/images/glotlid_logo.svg" alt="GlotLID" width="30%" />
</p>
<p align="center">
<a href="https://huggingface.co/cis-lmu/glotlid"><img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-8A2BE2"></a>
<a href="https://huggingface.co/spaces/cis-lmu/glotlid-space"><img alt="HuggingFace Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space (Demo)-orange"></a>
<a href="https://huggingface.co/datasets/cis-lmu/glotlid-corpus"><img alt="HuggingFace Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue"></a>
<a href="."><img alt="GitHub stars" src="https://img.shields.io/github/stars/cisnlp/GlotLID"></a>
<a href="https://arxiv.org/abs/2310.16248"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.16248-b31b1b.svg"></a>
</p>

## Language Identification with Support for More Than 2000 Labels

**TL;DR**: The repository introduces **GlotLID**, an open-source language identification (LID) model with support for more than **2000 labels**.

**Latest:** GlotLID **v4 alpha** (`model_v4_alpha.bin`) is out as a pre-release of the next version: 2056 labels, a smaller model (783 MB instead of 1.69 GB), trained with [fastText-turbo](https://github.com/CommonGlot/fastText-turbo). It is not final yet; its labels and scores may still change, so `model.bin` stays v3 for now. See [versions](#versions).

**Stable:** GlotLID V3 supports 2102 labels (three-letter [ISO 639-3](https://iso639-3.sil.org/code_tables/639/data) codes with script). For more details on the supported languages and performance, as well as significant changes from previous versions, please refer to [languages-v3.md](./languages-v3.md).

## Features
- Language Identification
- Get Sentence Vectors in respect to many languages
- Limit the language identification model to a smaller set of languages: The SET! evaluation part in the original paper.
- Code-switching: find all the languages of a mixed-language text, and the words of each, with [MaskLID](https://github.com/cisnlp/MaskLID) (try it in the Code-Switching tab of the [demo](https://huggingface.co/spaces/cis-lmu/glotlid-space)).

## How to use

### Load the model (Python)

```python
# pip install fasttext-turbo huggingface_hub

import fasttext
from huggingface_hub import hf_hub_download

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)
print("model path:", model_path)

# load the model
model = fasttext.load_model(model_path)
```

We recommend [fastText-turbo](https://github.com/CommonGlot/fastText-turbo) (`pip install fasttext-turbo`): it is a drop-in replacement for fastText (same `import fasttext`, same model files and API, prebuilt wheels) that predicts with GlotLID about 3 times faster than the original fastText in our tests (2.7x with v4 alpha, 3.4-4.3x with v3), with the same predictions. The original `fasttext` package (or `fasttext-numpy2-wheel`) works too.

If you do not want to always get the latest model (`model.bin`) and prefer to fix the version you are using, use `model_v?.bin` for example `model_v3.bin` (`?` is the version: we have `1`, `2`, `3`), or `model_v4_alpha.bin` for the v4 pre-release. See [versions](./#versions) for more.

Now you can use it for different applications:

#### For Language Identification :

```python
"""Language Identification"""
# predict language label (call this function as many times as needed)
model.predict("Hello, world!")
# (('__label__eng_Latn',), array([0.99850202]))
```

If you need top k prediction then, for example k = 3:
```python
model.predict("Hello, world!", 3)
# (('__label__eng_Latn', '__label__sun_Latn', '__label__ind_Latn'), array([9.98502016e-01, 7.95848144e-04, 3.11827025e-04]))
```

If you want to see which labels (language_scripts) are supported by the model, check the list of them by:

```python
model.labels
# ['__label__eng_Latn', '__label__rus_Cyrl', '__label__arb_Arab', '__label__por_Latn', ...]
```

#### For getting sentence vectors:

```python
"""Sentence Vectors"""
# get sentence vector of input sentence (call this function as many times as needed)
embedding = model.get_sentence_vector(sent)
```

### Limit the model to a smaller set of languages (Python)

You do not need to train another model for this. Just limit the prediction to the set you want. You can do it in two ways: `after` the prediction over all available classes, limit to those you want, or `before` the prediction, first limit to those you want, and then predict. If you want your probability to also be normalized to the set you specify, choose the `before` mode (default). However, if you want the probability you have to be in respect to all languages and then become limited to the set you have chosen, use the `after` mode (a text in none of your languages then gets a low probability for all of them).

The `CustomLID` class is in [customlid.py](./assets/inference/customlid.py):

```python
from huggingface_hub import hf_hub_download
from customlid import CustomLID

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

# labels with or without "__label__"; check the supported labels in model.labels
limited_languages = ['eng_Latn', 'arb_Arab', 'rus_Cyrl', 'por_Latn', 'pol_Latn', 'hin_Deva']

model = CustomLID(model_path, languages=limited_languages, mode='before')

model.predict("Hello, world!", 3)
# (('__label__eng_Latn', '__label__pol_Latn', '__label__hin_Deva'), array([9.9995399e-01, 4.2381755e-05, 3.2946982e-06], dtype=float32))

# many texts at once (faster than one by one): one result per text
model.predict(["Hello, world!", "Привет, мир!"], 3)
# [(('__label__eng_Latn', ...), array([...])), (('__label__rus_Cyrl', ...), array([...]))]
```

- `model_path` can also be a model you already loaded with `fasttext.load_model`, so several `CustomLID`s can share one model in memory.
- `predict(text, k=1, threshold=0.0)` works like fastText's `predict`: `k=-1` returns all the chosen labels, and `threshold` drops labels with a lower probability. Line breaks in the text count as spaces.
- Labels that the model does not have are skipped with a warning. The labels keep the model's order, whatever the order of `languages`.

## Versions

We always maintain the previous version of GlotLID in our [huggingface](https://huggingface.co/cis-lmu/glotlid) repository.

To access a specific version, simply append the version number to the filename.

For v1: `model_v1.bin` (introduced in the GlotLID paper and used in all experiments).

For v2: `model_v2.bin` (an edited version of v1, featuring more languages, and cleaned from noisy corpora based on the analysis of v1).
- It supports 1802 three-letter iso codes (1847 three letter iso codes with script)
- For 1626 three-letter iso codes; v2 on the test set achieved F1 of 0.996 and FPR of 0.0002.
  - These 1626 languages are selected based on the 0.5 F1 threshold and 0.0005 FPR threshold for low resource languages.

For v3: `model_v3.bin` (an edited version of v2, featuring more languages, excluding macro languages, further cleaned from noisy corpora and incorrect metadata labels based on the analysis of v2, supporting "zxx" and "und" series labels).
- It supports 1880 three-letter ISO codes (2114 three-letter ISO codes with script).
- For more details on the supported languages and performance, significant changes from previous versions, refer to [languages-v3.md](./languages-v3.md).


For v4 alpha: `model_v4_alpha.bin` (a pre-release of v4; not final, its labels and scores may still change).
- It supports 2056 labels (1899 three-letter ISO codes, with the `zxx` and `und` series).
- Trained with [fastText-turbo](https://github.com/CommonGlot/fastText-turbo); it loads in fastText and fastText-turbo like the previous versions. It is smaller (96-dimensional, 783 MB) and uses character n-grams of 2 to 5 with word n-grams of 1, so nothing is shared between neighbouring words: the scores of a text add up from the scores of its words, which makes the per-word scores that [MaskLID](https://github.com/cisnlp/MaskLID) uses for code-switching exact.

`model.bin` always refers to the latest stable version (v3 now).


## Data Sources 

See list of data sources [here](./sources.md).
We made a version of the dataset available for research purposes, which can be requested for access: https://huggingface.co/datasets/cis-lmu/glotlid-corpus

You're welcome to open a [pull request](https://github.com/cisnlp/GlotLID/pulls) or ([issue](https://github.com/cisnlp/GlotLID/issues)) and contribute new resources to our data list. Even for the languages we already support, we're actively seeking additional resources to mitigate domain shift issues.


## Benchmark 

- UDHR: access our clean version of udhr [here](https://huggingface.co/datasets/cis-lmu/udhr-lid).
- FLORES-200: devtest part of [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

## Evaluation

You can find how we compute F1, Recall, Precision, and FPR in [metrics.py](./assets/inference/metrics.py).


## FAQ
- If you see wrong predicted tags by GlotLID for a normal long text open an [issue](https://github.com/cisnlp/GlotLID/issues), however:
  - if the script is not supported by our model then use [GlotScript](https://github.com/cisnlp/GlotScript) to verify for the predicted `lang_script`, script in the sentence exists!  Otherwise, you need to write a function that returns 'und_mainscript' in this situations. GlotScript can identify both the mainscript and all available scripts in the sentence. We recommend using GlotLID in conjunction with GlotScript.
  - The high confidence threshold for each language could be different. This is because not all languages have the same distance from each other. For one language, 0.6 is a lot because it is very close to a similar language (such as dyu and bam), while for another, 0.9 might not be. 
  - This model is primarily trained on longer sentences, avoid using it on very short sentences. Other language identification models are not good at short sentences as well unless you increase the ngram size (in training), which is computationally expensive.
  -  In GlotLID, the false positive rate (FPR) for high-resource languages is much higher than for low-resource languages. However, even with this higher FPR, it is still lower than in a situation where the language identification model only recognizes high-resource languages. We are also okay with this situation since our main concern is for the FPR of low-resource languages to be low. The high-resource base frequency is much higher than for low-resource languages, so cleanliness would not be a threat for those languages. However, for a low-resource language with a low base frequency, even a small FPR might result in most of the corpus being noisy.
- Don't forget, you don't always need to run the language identification model in full. If you have a setup where you know it can only be a set of specific languages (subset of languages), then limit the prediction to those. We have provided the code for limiting it in this readme.
- If you want to add a language, provide the resource in an open [issue](https://github.com/cisnlp/GlotLID/issues), and we will add it. If you require the model urgently, we can expedite the process in less than a week (the training itself takes less than a day). However, if there's no immediate urgency, that language will be included in the official release according to our schedule (depends on new resources).-
- If you want to collaborate, please send us an email (to: amir@cis.lmu.de) specifying the type of collaboration you need from us.
- for the rest of requests feel free to email or open an issue.

## License
The model is distributed under the Apache License, Version 2.0 plus notices (see LICENSE file for full terms).

- Additional Notice:
This software includes a machine learning model for language identification based on statistical representations (e.g., character n-grams and linear classification).
The model was trained on a mixture of publicly available data sources, which may
include materials subject to copyright or other rights.
The model does not store or encode expressive text content. Instead, it represents aggregated statistical patterns (such as n-gram frequency weights) that are not
intended to reproduce original source material.
No training data is included or distributed with this software. 
Due to its design, this model performs classification only and is not capable of generating or reconstructing source texts.

- Good-Faith Notice:
If you are a rights holder and believe that your content has been included in the
training data and wish to request its exclusion from future versions, please contact
the authors. Reasonable efforts will be made to review and address such requests.


## Citation

If you find our model, code and list of data sources useful for your research, please cite:

```
@inproceedings{
    kargaran2023glotlid,
    title={{G}lot{LID}: Language Identification for Low-Resource Languages},
    author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
    booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023},
    url={https://openreview.net/forum?id=dl4e3EBz5j}
}
```

