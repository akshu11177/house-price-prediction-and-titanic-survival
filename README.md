# Machine Learning Assignments: Housing + Titanic

This submission-ready project contains:
1. **Housing price prediction — Linear Regression** (Assignment 1)
2. **Titanic survival prediction — Logistic Regression** (Assignment 2)
3. **House Price Prediction Model** (Mini Project 2), including R² evaluation and an actual-vs-predicted plot.

The house-price assignment and mini-project use the same Kaggle House Prices dataset and model workflow, avoiding unnecessary duplication.

## Privacy and data handling
- No personal details, account credentials, API keys, passwords, or private files are included.
- The ZIP contains source code and instructions, **not the Kaggle CSV datasets**.
- After downloading the datasets, keep the CSVs on your own computer in the folders shown below. The scripts process them locally and do not upload data anywhere.

## 1. Project structure

```text
House_Price_Prediction_and_ML_Assignments/
├── data/
│   ├── house_prices/       # Put Kaggle House Prices train.csv here
│   └── titanic/            # Put Titanic train.csv here
├── notebooks/
│   ├── house_price_prediction.ipynb
│   └── titanic_logistic_regression.ipynb
├── outputs/                # Charts and metrics are saved here after running
├── src/
│   ├── house_price_model.py
│   └── titanic_model.py
├── requirements.txt
├── run_all.py
└── README.md
```

## 2. Download the datasets

### House Prices — Advanced Regression Techniques
Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Download `train.csv` and place it at:
`data/house_prices/train.csv`

The target column is `SalePrice`.

### Titanic
Kaggle: https://www.kaggle.com/competitions/titanic/data

Download `train.csv` and place it at:
`data/titanic/train.csv`

The target column is `Survived`.

> Kaggle may require a free account and acceptance of the dataset/competition rules. The CSV files are intentionally not redistributed in this ZIP.

## 3. Install and run

Use Python 3.10 or newer.

### Windows (PowerShell or Command Prompt)

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

You can also run each model separately:

```bash
python src/house_price_model.py
python src/titanic_model.py
```

## 4. Outputs

After successful execution, files are saved in `outputs/`:

- `house_price_actual_vs_predicted.png` — scatter plot comparing actual and predicted house prices, with a reference diagonal.
- `house_price_metrics.txt` — R², MAE, and RMSE for the held-out test set.
- `titanic_confusion_matrix.png` — confusion matrix for survival classification.
- `titanic_metrics.txt` — accuracy, precision, recall, F1-score, and ROC-AUC (when calculable).

Metrics are calculated from a reproducible train/test split. The R² score is measured on the held-out house-price test set; it is not guaranteed to be a particular value and can vary slightly with library versions.

## 5. What the project does

### House price prediction
- Separates `SalePrice` from the input features.
- Removes the identifier column `Id` when present.
- Splits data into training and testing sets.
- Imputes missing numerical values with the median and categorical values with the most frequent value.
- One-hot encodes categorical columns.
- Trains a scikit-learn Linear Regression model.
- Predicts prices for the test set.
- Evaluates R², MAE, and RMSE.
- Saves an actual-vs-predicted plot.

### Titanic survival prediction
- Separates `Survived` from the input features.
- Removes the passenger identifier `PassengerId` when present.
- Handles missing values and categorical features in a preprocessing pipeline.
- Trains a Logistic Regression classifier.
- Evaluates predictions using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC where possible.

## 6. Notebooks

Open the notebooks in Jupyter after installing requirements:

```bash
jupyter notebook
```

Then open the relevant `.ipynb` file in the `notebooks/` folder. The notebooks call the same tested project scripts so the code stays consistent.

## 7. Troubleshooting

- **File not found:** Confirm that the downloaded file is named exactly `train.csv` and is in the correct dataset folder.
- **Missing package:** Run `pip install -r requirements.txt` in the activated environment.
- **Different column names:** These scripts are designed for the standard Kaggle competition training CSVs described above.
- **Do not upload private information:** Only the two public dataset CSVs are needed for this project. No personal profile details are required.

## Submission note

Submit this ZIP along with the two Kaggle CSV files only if your evaluator explicitly requires the raw data. Otherwise, submit the ZIP and mention the dataset source links in this README. Follow your institution's instructions for dataset licensing and submission.
