"""Dataset loaders, one module per dataset.

Rule (approval): data-dependent preprocessing (scaling, one-hot encoding) is fit on the
training split only.
Guideline: include data only when its license and size permit; otherwise provide download
links or scripts and document every preprocessing step.

Loader interface (docs/open-questions.md #11): ``load_xxx(split_seed=4330)`` returns
``X_train, y_train, X_val, y_val, X_test, y_test`` as NumPy arrays; all feature preprocessing is
done inside the loader. Raw files are downloaded into data/ on first use (``fetch``).
"""

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def fetch(url, filename):
    """Return the path data/<filename>, downloading ``url`` there first if it is missing."""
    path = DATA_DIR / filename
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path
