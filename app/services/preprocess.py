import re
import unicodedata

STOPWORDS = frozenset(
    {
        "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas", "ao", "aos", "para", "por", "pelo", "pela", "com",
        "e", "ou", "que", "se", "foi", "esta", "estao", "ate", "ha", "mas", "ja", "ele",
        "ela", "seu", "sua", "este", "essa", "isso", "aqui", "la", "ser",
    }
)

_CURRENT_LOOP = re.compile(r"\b4\s*(?:-|a)?\s*20\s*ma\b")
_ERROR_CODE = re.compile(r"\b([ef])-?0*(\d{1,3})\b")
_NON_TOKEN = re.compile(r"[^a-z0-9_\s]")
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def fold_text(text: str) -> str:
    return _SPACES.sub(" ", strip_accents(text.lower())).strip()


def normalize_text(text: str) -> str:
    folded = strip_accents(text.lower())
    folded = _CURRENT_LOOP.sub(" sinal_4_20ma ", folded)
    folded = _ERROR_CODE.sub(r" cod_\1\2 ", folded)
    folded = _NON_TOKEN.sub(" ", folded)
    tokens = [token for token in _SPACES.split(folded) if token and token not in STOPWORDS]
    return " ".join(tokens)
