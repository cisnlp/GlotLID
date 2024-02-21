import fasttext
import numpy as np
from huggingface_hub import hf_hub_download

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



# download model
## cache_dir: path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)


# to make sure these languages are available in GlotLID check the list of supported labels in model.labels
limited_languages = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__hin_Deva']

model = CustomLID(model_path, languages = limited_languages , mode='before')

model.predict("Hello, world!", 3)
