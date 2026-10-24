# DecodeLabs — AI Project 2: Data Classification Using AI

A beginner-friendly supervised machine-learning project built to match the DecodeLabs Project 2 brief.

## Project objective
Build a basic classification model using a small dataset. The project demonstrates how to load and understand data, split it into training and testing sets, train a classification algorithm, and evaluate predictions.

## Dataset
The included `data/iris.csv` contains 150 Iris flower samples with four measurements:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target column is `species` (`setosa`, `versicolor`, or `virginica`). The data is included locally, so no download is required to run the project.

## Algorithm and workflow
1. Load the CSV with pandas and inspect rows, shape, class counts, and missing values.
2. Separate features (`X`) from the target (`y`).
3. Split data into 80% training and 20% testing data with stratification and a fixed random seed.
4. Scale features using `StandardScaler`.
5. Train a `LogisticRegression` classifier in a scikit-learn pipeline.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Predict the species for an example flower measurement.

## Requirements
- Python 3.9 or newer
- pip

## Run locally

### 1. Create and activate a virtual environment (recommended)
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python main.py
```

The script prints dataset information and evaluation metrics. It creates:
- `outputs/metrics.json`
- `outputs/sample_prediction.json`
- `outputs/confusion_matrix.png`

## Project structure
```text
DecodeLabs_AI_Classification_Project/
├── data/
│   └── iris.csv
├── outputs/
│   └── .gitkeep
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Results
The script calculates the score each time it runs; the exact result is saved to `outputs/metrics.json`. Because the test set is held out from training, the metrics provide a basic estimate of generalization, not a guarantee of performance on every future dataset.

## GitHub upload
Create a new empty repository on GitHub, then from this project folder run:
```bash
git init
git add .
git commit -m "Complete DecodeLabs AI classification project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the URL with your own repository URL. Do not upload passwords, access tokens, or other secrets.

## Learning outcomes
- Data loading and inspection
- Train/test split
- Supervised learning
- Logistic regression classification
- Model evaluation and sample prediction

---
Prepared as a practical implementation of the requirements described in the DecodeLabs Artificial Intelligence Project 2 training brief.
