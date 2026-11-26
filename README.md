# Classifying Music Genres Using Machine Learning

CIS8695 · Group 4

This project compares an RBF support vector machine and a random forest on FMA-small. Both models use precomputed audio features to predict one of eight genres. We select hyperparameters using validation macro F1, compare both models against a majority-class baseline, and report accuracy, macro F1, per-genre metrics, and confusion matrices on the held-out test set. Grouped permutation importance measures which audio feature families each model relies on.

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

Open `notebooks/music_genre_classification.ipynb` and run all cells. The two CSVs contain the labels, split assignments, and predictors needed for the notebook. The exported results were generated with Python 3.10.6 and scikit-learn 1.7.2.

## Listening demo

The last notebook cell uses three held-out tracks to show a correct prediction, a disagreement, and a shared mistake. Open each card to reveal the archive label and both predictions. For playback, extract the corresponding files from the FMA-small audio archive into their original paths under `data/fma_small/`:

- `001/001075.mp3`
- `000/000704.mp3`
- `000/000182.mp3`

Run the notebook locally with those clips in place. Audio can be embedded in notebook outputs, so save the version with playback under the ignored `data/` folder.

## Project files

- `notebooks/` contains the analysis and model comparison.
- `results/` holds the exported metrics and charts used in the slides and report.
- `deliverables/` holds final presentation and report files.

The [metrics table](results/metrics.csv) compares the two models and the baseline. The [feature importance chart](results/feature_family_importance.png) shows how much validation macro F1 drops when each family of audio features is shuffled.

Dataset: [Free Music Archive (FMA)](https://github.com/mdeff/fma). See [Defferrard et al. (2017)](https://arxiv.org/abs/1612.01840).
