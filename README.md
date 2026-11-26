# Classifying Music Genres Using Machine Learning

CIS8695 · Group 4

This project compares an RBF support vector machine and a random forest on FMA-small. Both models use precomputed audio features to predict one of eight genres. We select hyperparameters using validation macro F1 and report accuracy, macro F1, per-genre metrics, and confusion matrices on the held-out test set.

## Run locally

Use Python 3.10 or later. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl -fL https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -o data/fma_metadata.zip
python -m zipfile -e data/fma_metadata.zip data
python -m jupyterlab
```

Open `notebooks/music_genre_classification.ipynb` and run all cells. The original MP3 archive is optional. The two CSVs already contain the labels, split assignments, and predictors.

## Project files

- `notebooks/` contains the analysis and model comparison.
- `results/` holds the exported metrics and charts used in the slides and report.
- `deliverables/` holds final presentation and report files.

Start from the notebook's validation results, then use the test results and confusion matrices as evidence in both the presentation and the report.

Dataset: [Free Music Archive (FMA)](https://github.com/mdeff/fma). See [Defferrard et al. (2017)](https://arxiv.org/abs/1612.01840).
