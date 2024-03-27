
<p align="center">
<img src="./assets/images/glotlid_logo.svg" alt="GlotLID" width="30%" />
</p>
<p align="center">
<a href="https://huggingface.co/cis-lmu/glotlid"><img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-8A2BE2"></a>
<a href="https://huggingface.co/spaces/cis-lmu/glotlid-space"><img alt="HuggingFace Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space (Demo)-orange"></a>
<a href="https://github.com/cisnlp/GlotLID/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/cisnlp/GlotLID?logoColor=blue"></a>
<a href="."><img alt="GitHub stars" src="https://img.shields.io/github/stars/cisnlp/GlotLID"></a>
<a href="https://arxiv.org/abs/2310.16248"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.16248-b31b1b.svg"></a>
</p>

## Language Identification with Support for More Than 2000 Labels

**TL;DR**: The repository introduces **GlotLID**, an open-source language identification (LID) model with support for more than **2000 labels**.

**Latest:** GlotLID is now updated to V3. V3 supports 2102 labels (three-letter [ISO 639-3](https://iso639-3.sil.org/code_tables/639/data) codes with script). For more details on the supported languages and performance, as well as significant changes from previous versions, please refer to [languages-v3.md](./languages-v3.md).

## Features
- Language Identification
- Get Sentence Vectors in respect to many languages
- Limit the language identification model to a smaller set of languages: The SET! evaluation part in the original paper.

## How to use

### Load the model (Python)

```python
# pip install fasttext
# pip install huggingface_hub

import fasttext
from huggingface_hub import hf_hub_download

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)
print("model path:", model_path)

# load the model
model = fasttext.load_model(model_path)
```

If you do not want to always get the latest model (`model.bin`) and prefer to fix the version you are using, use `model_v?.bin` for example `model_v3.bin` (`?` is the version: we have `1`, `2`, `3`). See [versions](./#versions) for more.

Now you can use it for differnet applications:

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

You do not need to train another model for this. Just limit the prediction to the set you want. You can do it in two ways: `after` the prediction over all available classes, limit to those you want, or `before` the prediction, first limit to those you want, and then predict. If you want your probability to also be normalized to the set you specify, choose the `before` mode (default). However, if you want the probability you have to be in respect to all languages and then become limited to the set you have chosen, use the `after` mode.

You can find how we to do this in [customlid.py](./assets/inference/customlid.py).

<details>
<summary>Or click to see the code here.</summary>


```python
import fasttext
import numpy as np
class CustomLID:
    def __init__(self, model_path, languages = -1, mode='before'):
        self.model = fasttext.load_model(model_path)
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()
        
        # compute language_indices
        if languages !=-1 and isinstance(languages, list):
            self.language_indices = [self.labels.index(l) for l in list(set(languages)) if l in self.labels]

        else:
            self.language_indices = list(range(len(self.labels)))

        # limit labels to language_indices
        self.labels = list(np.array(self.labels)[self.language_indices])
        
        # predict
        self.predict = self.predict_limit_after_softmax if mode=='after' else self.predict_limit_before_softmax

    
    def predict_limit_before_softmax(self, text, k=1):
        
        # sentence vector
        sentence_vector = self.model.get_sentence_vector(text)
        
        # dot
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)

        # softmax
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(np.exp(result_vector - np.max(result_vector)))

        # top k predictions
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs


    def predict_limit_after_softmax(self, text, k=1):
        
        # sentence vector
        sentence_vector = self.model.get_sentence_vector(text)
        
        # dot
        result_vector = np.dot(self.output_matrix, sentence_vector)

        # softmax
        softmax_result = np.exp(result_vector - np.max(result_vector)) / np.sum(np.exp(result_vector - np.max(result_vector)))

        # limit softmax to language_indices
        softmax_result = softmax_result[self.language_indices]

        
        # top k predictions
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]

        return tuple(top_k_labels), top_k_probs

```

You can load the model with CustomLID class to limit your prediction to the set of limited_languages:

```python
from huggingface_hub import hf_hub_download

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)
print("model path:", model_path)


# to make sure these languages are available in GlotLID check the list of supported labels in model.labels
limited_languages = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__hin_Deva']

model = CustomLID(model_path, languages = limited_languages , mode='before')

model.predict("Hello, world!", 3)
```

</details>

## Versions

We always maintain the previous version of GlotLID in our [huggingface](https://huggingface.co/cis-lmu/glotlid) repository.

To access a specific version, simply append the version number to the filename.

For v1: `model_v1.bin` (introduced in the GlotLID paper and used in all experiments).

For v2: `model_v2.bin` (an edited version of v1, featuring more languages, and cleaned from noisy corpora based on the analysis of v1).
- It suuports 1802 three-letter iso codes (1847 three letter iso codes with script)
- For 1626 three-letter iso codes; v2 on the test set achieved F1 of 0.996 and FPR of 0.0002.
  - These 1626 languages are selected based on the 0.5 F1 threshold and 0.0005 FPR threshold for low resource languages.

For v3: `model_v3.bin` (an edited version of v2, featuring more languages, excluding macro languages, further cleaned from noisy corpora and incorrect metadata labels based on the analysis of v2, supporting "zxx" and "und" series labels).
- It supports 1880 three-letter ISO codes (2114 three-letter ISO codes with script).
- For more details on the supported languages and performance, significant changes from previous versions, refer to [languages-v3.md](./languages-v3.md).


`model.bin` always refers to the latest version (v3 now).


## Data Sources 

See list of data sources [here](./sources.md).

You're welcome to open a [pull request](https://github.com/cisnlp/GlotLID/pulls) or ([issue](https://github.com/cisnlp/GlotLID/issues)) and contribute new resources to our data list. Even for the languages we already support, we're actively seeking additional resources to mitigate domain shift issues.


## Benchmark 

- UDHR: access our clean version of udhr [here](https://huggingface.co/datasets/cis-lmu/udhr-lid).
- FLORES-200: devtest part of [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

## Evaluation

You can find how we compute F1, Recall, Precision, and FPR in [metrics.py](./assets/inference/metrics.py).


## FAQ
- If you see wrong predicted tags by GlotLID for a normal long text open an [issue](https://github.com/cisnlp/GlotLID/issues), however:
  - if the script is not supported by our model then use [GlotScript](https://github.com/cisnlp/GlotLID) to verify for the predicted `lang_script`, script in the sentence exists!  Otherwise, you need to write a function that returns 'und_mainscript' in this situations. GlotScript can identify both the mainscript and all available scripts in the sentence. We recommend using GlotLID in conjunction with GlotScript.
  - The high confidence threshold for each language could be different. This is because not all languages have the same distance from each other. For one language, 0.6 is a lot because it is very close to a similar language (such as dyu and bam), while for another, 0.9 might not be. 
  - This model is primarily trained on longer sentences, avoid using it on very short sentences. Other language identification models are not good at short sentences as well unless you increase the ngram size (in training), which is computationally expensive.
  -  In GlotLID, the false positive rate (FPR) for high-resource languages is much higher than for low-resource languages. However, even with this higher FPR, it is still lower than in a situation where the language identification model only recognizes high-resource languages. We are also okay with this situation since our main concern is for the FPR of low-resource languages to be low. The high-resource base frequency is much higher than for low-resource languages, so cleanliness would not be a threat for those languages. However, for a low-resource language with a low base frequency, even a small FPR might result in most of the corpus being noisy.
- Don't forget, you don't always need to run the language identification model in full. If you have a setup where you know it can only be a set of specific languages (susbet of languages), then limit the prediction to those. We have provided the code for limiting it in this readme.
- If you want to add a language, provide the resource in an open [issue](https://github.com/cisnlp/GlotLID/issues), and we will add it. If you require the model urgently, we can expedite the process in less than a week (the training itself takes less than a day). However, if there's no immediate urgency, that language will be included in the official release according to our schedule (depends on new resources).-
- If you want to collaborate, please send us an email (to: amir@cis.lmu.de) specifying the type of collaboration you need from us.
- for the rest of requests feel free to email or open an issue.

## Citation

If you find our model, code and list of data sources useful for your research, please cite:

```python
@inproceedings{
  kargaran2023glotlid,
  title={GlotLID: Language Identification for Low-Resource Languages},
  author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  url={https://openreview.net/forum?id=dl4e3EBz5j}
}
```


