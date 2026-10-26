"""Download all three datasets into data/.

Guideline: the repository must contain data acquisition and preprocessing instructions.
Run: python scripts/download_data.py   (files already present are skipped)

- Fashion-MNIST: the official IDX files from https://github.com/zalandoresearch/fashion-mnist (MIT)
- Adult Census Income: adult.data and adult.test from the UCI ML Repository (CC BY 4.0)
- California Housing: scikit-learn's copy of the StatLib data, cached in data/
Preprocessing is done by the loaders in src/bench/data/ (fit on the training split only).
"""

import urllib.request

from sklearn.datasets import fetch_california_housing

from bench.data import DATA_DIR

FMNIST = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
ADULT = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES = [
    *(FMNIST + f for f in (
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    )),
    ADULT + "adult.data",
    ADULT + "adult.test",
]

DATA_DIR.mkdir(exist_ok=True)
for url in FILES:
    path = DATA_DIR / url.rsplit("/", 1)[1]
    if path.exists():
        print(f"have {path.name}")
    else:
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
fetch_california_housing(data_home=DATA_DIR)
print(f"all datasets are in {DATA_DIR}")
