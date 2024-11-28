import gcld3
from GlotScript import get_script_predictor

sp = get_script_predictor()

class CLD3:

    def __init__(self, conf=0.0, type_=""):
        # Load model
        self.model = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000000)
        self.conf = conf
        self.cld3_map = {
          "af": "afr-Latn",
          "am": "amh-Ethi",
          "ar": "ara-Arab",
          "bg": "bul-Cyrl",
          "az": "azj-Latn",
          "be": "bel-Cyrl",
          "bg-Latn": "bul-Latn",
          "bn":  "ben-Beng",
          "bs":  "bos-Latn",
          "ca":  "cat-Latn",
          "ceb": "ceb-Latn",
          "co": "cos-Latn",
          "cs": "ces-Latn",
          "cy": "cym-Latn",
          "da": "dan-Latn",
          "de": "deu-Latn",
          "el": "ell-Grek",
          "el-Latn": "ell-Latn",
          "en": "eng-Latn",
          "eo": "epo-Latn",
          "es": "spa-Latn",
          "et": "est-Latn",
          "eu": "eus-Latn",
          "fa": "fas-Arab",
          "fi": "fin-Latn",
          "fil": "fil-Latn",
          "fr": "fra-Latn",
          "fy": "fry-Latn",
          "ga": "gle-Latn",
          "gd": "gla-Latn",
          "gl": "glg-Latn",
          "gu": "guj-Gujr",
          "ha": "hau-Latn",
          "haw": "haw-Latn",
          "hi": "hin-Deva",
          "hi-Latn": "hin-Latn",
          "hmn": "hmn-Latn",
          "hr": "hrv-Latn",
          "ht": "hat-Latn",
          "hu": "hun-Latn",
          "hy": "hye-Armn",
          "id": "msa-Latn", # because of ms
          "ig": "ibo-Latn",
          "is": "isl-Latn",
          "it": "ita-Latn",
          "iw": "heb-Hebr",
          "ja": "jpn-Jpan",
          "ja-Latn": "jpn-Latn",
          "jv": "jav-Latn",
          "ka": "kat-Geor",
          "kk": "kaz-Cyrl",
          "km": "khm-Khmr",
          "kn": "kan-Knda",
          "ko": "kor-Hang",
          "ku": "kmr-Latn",
          "ky": "kir-Cyrl",
          "la": "lat-Latn",
          "lb": "ltz-Latn",
          "lo": "lao-Laoo",
          "lt": "lit-Latn",
          "lv": "lav-Latn",
          "mg": "mlg-Latn",
          "mi": "mri-Latn",
          "mk": "mkd-Cyrl",
          "ml": "mal-Mlym",
          "mn": "mon-Cyrl",
          "mr": "mar-Deva",
          "ms": "msa-Latn",
          "mt": "mlt-Latn",
          "my": "mya-Mymr",
          "ne": "nep-Deva",
          "nl": "nld-Latn",
          "no": "nor-Latn",
          "ny": "nya-Latn",
          "pa": "pan-Guru",
          "pl": "pol-Latn",
          "ps": "pus-Arab",
          "pt": "por-Latn",
          "ro": "ron-Latn",
          "ru": "rus-Cyrl",
          "ru-Latn": "rus-Latn",
          "sd": "snd-Arab",
          "si": "sin-Sinh",
          "sk": "slk-Latn",
          "sl": "slv-Latn",
          "sm": "smo-Latn",
          "sn": "sna-Latn",
          "so": "som-Latn",
          "sq": "sqi-Latn",
          "sr": "srp-Cyrl",
          "st": "sot-Latn",
          "su": "sun-Latn",
          "sv": "swe-Latn",
          "sw": "swa-Latn",
          "ta": "tam-Taml",
          "te": "tel-Telu",
          "tg": "tgk-Cyrl",
          "th": "tha-Thai",
          "tr": "tur-Latn",
          "uk": "ukr-Cyrl",
          "ur": "urd-Arab",
          "uz": "uzb-Latn",
          "vi": "vie-Latn",
          "xh": "xho-Latn",
          "yi": "yid-Hebr",
          "yo": "yor-Latn",
          "zh": "zho-Hani",
          "zh-Latn": "zho-Latn",
          "zu": "zul-Latn"
          }
        
    def predict_lang_with_confidence(self, text: str) -> Tuple[str, float]:
        # Predict language label and confidence
        detect = self.model.FindLanguage(text)
        
        if detect.probability < self.conf:
            script = sp(text)[0]
            return f"und_{script}", 0.0
        
        pred_iso_part3 = self.cld3_map[detect.language]        
        
        return pred_iso_part3, detect.probability

