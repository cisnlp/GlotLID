"""CustomLID: GlotLID limited to a set of languages, without training another model.

A fastText model is linear: the sentence vector times the output matrix gives one score per
label, and the probabilities are their softmax. To limit the model to some languages, either
take the softmax over the scores of those languages only (mode="before", the default: the
probabilities sum to 1 over the set) or take it over all labels and keep the chosen ones
(mode="after": each probability is the one the full model gives, so a text in none of the
chosen languages gets low probabilities for all of them).

  pip install fasttext-turbo huggingface_hub    # fasttext-turbo installs the `fasttext` module

  from customlid import CustomLID
  model = CustomLID(model_path, languages=["eng_Latn", "arb_Arab", "rus_Cyrl"])
  model.predict("Hello, world!", k=3)
  model.predict(["Hello, world!", "Привет, мир!"])   # a list: one result per text
"""
import contextlib
import warnings

import fasttext
import numpy as np

try:
    from threadpoolctl import ThreadpoolController
except ImportError:  # optional: without it NumPy uses its default number of BLAS threads
    ThreadpoolController = None

_CONTROLLER = None


def _single_blas_thread():
    """Run NumPy's BLAS on one thread: these products are small, and a multithreaded BLAS spends
    longer starting threads than computing them."""
    global _CONTROLLER
    if ThreadpoolController is None:
        return contextlib.nullcontext()
    if _CONTROLLER is None:
        _CONTROLLER = ThreadpoolController()
    return _CONTROLLER.limit(limits=1, user_api="blas")


class CustomLID:
    def __init__(self, model_path, languages=-1, mode="before"):
        """
        Args:
            model_path (str or fastText model): path to a GlotLID model, or a model already
                loaded with fasttext.load_model (used as is, not copied).
            languages (list or -1): the labels to keep, with or without the "__label__" prefix
                (e.g. "eng_Latn"); -1 keeps all labels. Labels the model does not have are
                skipped with a warning.
            mode (str): "before" (softmax over the chosen labels) or "after" (softmax over all
                labels, then the chosen ones).
        """
        if mode not in ("before", "after"):
            raise ValueError('mode must be "before" or "after"')
        self.model = fasttext.load_model(model_path) if isinstance(model_path, str) else model_path
        self.mode = mode
        self.output_matrix = self.model.get_output_matrix()
        all_labels = self.model.get_labels()

        if languages != -1 and isinstance(languages, (list, tuple, set)):
            wanted = {l if l.startswith("__label__") else "__label__" + l for l in languages}
            missing = sorted(wanted.difference(all_labels))
            if missing:
                warnings.warn("not in the model, skipped: " + ", ".join(missing), stacklevel=2)
            # in model order, so the result does not depend on the order of `languages`
            self.language_indices = [i for i, l in enumerate(all_labels) if l in wanted]
            if not self.language_indices:
                raise ValueError("none of the languages is in the model")
        else:
            self.language_indices = list(range(len(all_labels)))
        self.labels = [all_labels[i] for i in self.language_indices]
        # rows of the chosen labels, taken once
        self._limited = np.ascontiguousarray(self.output_matrix[self.language_indices])

    def _probs(self, vectors):
        """Probabilities of the chosen labels for sentence vectors (texts x dim, or one vector)."""
        if self.mode == "before":
            z = vectors @ self._limited.T
        else:
            z = vectors @ self.output_matrix.T
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        p = z / z.sum(axis=-1, keepdims=True)
        return p if self.mode == "before" else p[..., self.language_indices]

    def _vector(self, text):
        # fastText reads one line: line breaks count as spaces
        return self.model.get_sentence_vector(text.replace("\r", " ").replace("\n", " "))

    def _top(self, p, k, threshold):
        k = min(k, len(p)) if k > 0 else len(p)
        if k < len(p) and len(p) > 64:
            top = np.argpartition(-p, k - 1)[:k]               # the k best without sorting all labels
            top = top[np.argsort(-p[top], kind="stable")]
        else:
            top = np.argsort(-p, kind="stable")[:k]
        top = top[p[top] >= threshold]
        return tuple(self.labels[i] for i in top), p[top]

    def predict(self, text, k=1, threshold=0.0):
        """The k most probable of the chosen labels and their probabilities, as fastText's
        predict: (labels, probabilities). k=-1 gives all of them; threshold drops labels with a
        lower probability. Given a list of texts, returns a list of such results."""
        if isinstance(text, str):
            return self._top(self._probs(self._vector(text)), k, threshold)
        texts = list(text)
        if not texts:
            return []
        vectors = np.stack([self._vector(t) for t in texts])
        with _single_blas_thread():
            probs = self._probs(vectors)
        return [self._top(p, k, threshold) for p in probs]

    # The names of the original class, kept for code that calls them.
    def predict_limit_before_softmax(self, text, k=1):
        return CustomLID._with_mode(self, "before").predict(text, k)

    def predict_limit_after_softmax(self, text, k=1):
        return CustomLID._with_mode(self, "after").predict(text, k)

    def _with_mode(self, mode):
        if mode == self.mode:
            return self
        other = object.__new__(CustomLID)
        other.__dict__.update(self.__dict__, mode=mode)
        return other


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    # cache_dir: path to the folder where the downloaded model will be stored/cached.
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

    # check the supported labels with fasttext.load_model(model_path).labels
    limited_languages = ["eng_Latn", "arb_Arab", "rus_Cyrl", "por_Latn", "pol_Latn", "hin_Deva"]

    model = CustomLID(model_path, languages=limited_languages, mode="before")
    print(model.predict("Hello, world!", 3))
