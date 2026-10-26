"""Dataset loaders, one module per dataset.

Rule (approval): data-dependent preprocessing (scaling, one-hot encoding) is fit on the
training split only.
Guideline: include data only when its license and size permit; otherwise provide download
links or scripts and document every preprocessing step.

Loader interface (docs/open-questions.md #11): ``load_xxx(split_seed=4330)`` returns
``X_train, y_train, X_val, y_val, X_test, y_test`` as NumPy arrays; all feature preprocessing is
done inside the loader. Loaders only read the raw files in data/; ``scripts/download_data.py``
puts them there.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def data_path(filename):
    """Return data/<filename>, or tell the user to run the download script if it is missing."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python scripts/download_data.py")
    return path
