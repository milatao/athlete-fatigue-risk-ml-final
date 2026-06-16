# Athlete Fatigue Risk ML Final Project

Machine learning final project using nested cross-validation to compare classifiers for synthetic athlete fatigue risk prediction.

## Project overview

This project focuses on predicting athlete fatigue risk using supervised machine learning methods.

The task is formulated as a binary classification problem, where the target variable indicates whether an athlete is classified as normal or fatigued. Several classification models were compared, and the final model was selected based on the mean outer cross-validation F1-score.

The main goal of the project was to practice model comparison, hyperparameter tuning, final test-set evaluation and basic model interpretation.

## Dataset

This project uses the **AFR-1000 Athlete Fatigue Risk Dataset**, a synthetic dataset designed for athlete fatigue risk prediction and machine learning research.

Dataset source:  
[Yuanchun Hong / DataMaverick — AFR-1000 Athlete Fatigue Risk Dataset, Kaggle](https://www.kaggle.com/datasets/yuanchunhong/afr-1000-athlete-fatigue-risk-dataset)

License: CC0: Public Domain

Because the dataset is synthetic, the results should be interpreted mainly as a comparison of machine learning methods, not as direct clinical or sports-science evidence about real athletes.

## Methods

The following models were compared:

- Dummy Classifier
- Logistic Regression
- k-Nearest Neighbors
- Support Vector Machine
- Random Forest
- Multi-Layer Perceptron

Model comparison was performed using nested stratified cross-validation. The inner cross-validation loop was used for hyperparameter tuning, while the outer loop was used to estimate model performance.

The final model was selected according to the mean outer cross-validation F1-score. After model selection, the selected algorithm was tuned on the full training set and evaluated once on the independent held-out test set.

## Results

Logistic Regression achieved the best mean outer cross-validation F1-score and was selected as the final model.

On the held-out test set, the selected model achieved an F1-score of 0.985 and made only three incorrect predictions.

Permutation importance and Logistic Regression coefficients were used for post-hoc interpretation of the selected model. The most important features were mainly related to recovery and physiological stress, including HRV, cortisol level, sleep quality, heart rate recovery, perceived exertion and muscle soreness.

## Repository structure

```text
athlete-fatigue-risk-ml-final/
├── README.md
├── athlete_fatigue_project.py
├── data.csv
├── report.pdf
└── figures/
    ├── confusion_matrix.png
    ├── model_comparison_outer_cv_f1_std.png
    ├── final_test_permutation_importance.png
    └── final_model_logistic_regression_coefficients.png
```

## Technologies used

- Python
- NumPy
- pandas
- matplotlib
- scikit-learn

## How to run

Make sure that `data.csv` is located in the same folder as `athlete_fatigue_project.py`.

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Run the script:

```bash
python athlete_fatigue_project.py
```

The script loads `data.csv`, compares the models, evaluates the final model and generates output figures and a results summary.

## Output files

The script generates several output files, including:

- `results_summary.csv`
- `model_comparison_outer_cv_f1_std.png`
- `confusion_matrix.png`
- `final_test_permutation_importance.png`
- `final_model_logistic_regression_coefficients.png`

## Report

The full project report is available in:

```text
report.pdf
```

## Note

This project was created as a university machine learning final project. The results are unusually high because the dataset is synthetic and the target labels are likely based on predefined physiological rules.
