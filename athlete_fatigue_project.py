import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. Basic settings
# ============================================================

seed = 1
np.random.seed(seed)

csv_path = "data.csv"
target_column = "Fatigue_Risk"

# Main metric used for model selection.
main_scoring = "f1"


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(csv_path)

print("\nDATASET INFO")
print(df.info())

print("\nFIRST 5 ROWS")
print(df.head())

print("\nBASIC STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nTARGET DISTRIBUTION")
print(df[target_column].value_counts())
print(df[target_column].value_counts(normalize=True))


# ============================================================
# 3. Features and target
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]


# ============================================================
# 4. Hold-out train/test split
# ============================================================

# The test set is kept completely separate.
# It is used only once after model selection is finished.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    stratify=y
)

print("\nTRAIN/TEST SPLIT")
print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")


# ============================================================
# 5. Nested cross-validation setup
# ============================================================

# Outer CV estimates generalization performance during model comparison.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

# Inner CV tunes hyperparameters.
inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

scoring = {
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, zero_division=0),
}


# ============================================================
# 6. Models and parameter grids
# ============================================================

# The grids are intentionally smaller than in ordinary GridSearchCV,
# because nested CV is computationally more expensive.
models = {
    "Dummy Classifier": {
        "pipeline": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent", random_state=seed))
        ]),
        "params": {}
    },

    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=20000,
                random_state=seed
            ))
        ]),
        "params": {
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "params": {
            "clf__n_neighbors": [5, 11, 21, 31, 51],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan", "cosine"]
        }
    },

    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=seed))
        ]),
        "params": [
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__class_weight": [None, "balanced"]
            },
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.1, 1, 10, 100, 1000],
                "clf__gamma": ["scale", "auto", 0.0003, 0.001, 0.01, 0.1],
                "clf__class_weight": [None, "balanced"]
            }
        ]
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(
                random_state=seed
            ))
        ]),
        "params": {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 8, 10, 15],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None, "balanced"]
        }
    },

    "MLP": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                max_iter=5000,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=30
            ))
        ]),
        "params": {
            "clf__hidden_layer_sizes": [(50,), (100,), (100, 50), (150, 75), (100, 50, 25)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [0.00001, 0.0001, 0.001],
            "clf__learning_rate_init": [0.001, 0.003],
            "clf__solver": ["adam"]
        }
    }
}


# ============================================================
# 7. Nested CV model comparison
# ============================================================

nested_results = []
outer_fold_best_params = []

for model_name, model_data in models.items():
    print("\n" + "=" * 80)
    print(f"Nested CV evaluation: {model_name}")
    print("=" * 80)

    param_grid = model_data["params"] if model_data["params"] else [{}]

    inner_search = GridSearchCV(
        estimator=model_data["pipeline"],
        param_grid=param_grid,
        scoring=main_scoring,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )

    # cross_validate performs the outer CV.
    # In every outer fold, GridSearchCV is fitted only on the outer-training subset.
    # The outer validation fold is not used for hyperparameter tuning.
    cv_output = cross_validate(
        estimator=inner_search,
        X=X_train,
        y=y_train,
        cv=outer_cv,
        scoring=scoring,
        n_jobs=1,
        return_estimator=True
    )

    row = {
        "Model": model_name,
        "Outer CV accuracy mean": np.mean(cv_output["test_accuracy"]),
        "Outer CV accuracy std": np.std(cv_output["test_accuracy"]),
        "Outer CV F1 mean": np.mean(cv_output["test_f1"]),
        "Outer CV F1 std": np.std(cv_output["test_f1"]),
    }

    nested_results.append(row)

    print(f"Outer CV F1: {row['Outer CV F1 mean']:.4f} ± {row['Outer CV F1 std']:.4f}")

    print("\nBest parameters selected in each outer fold:")
    for fold_idx, estimator in enumerate(cv_output["estimator"], start=1):
        best_params = estimator.best_params_
        print(f"Fold {fold_idx}: {best_params}")

        outer_fold_best_params.append({
            "Model": model_name,
            "Outer fold": fold_idx,
            "Best parameters": best_params,
            "Inner CV best F1": estimator.best_score_,
            "Outer fold accuracy": cv_output["test_accuracy"][fold_idx - 1],
            "Outer fold F1": cv_output["test_f1"][fold_idx - 1],
        })


# ============================================================
# 8. Save nested CV results
# ============================================================

nested_results_df = pd.DataFrame(nested_results)
nested_results_df = nested_results_df.sort_values(
    by="Outer CV F1 mean",
    ascending=False
).reset_index(drop=True)

outer_fold_best_params_df = pd.DataFrame(outer_fold_best_params)

print("\n" + "=" * 80)
print("NESTED CV SUMMARY")
print("=" * 80)
print(nested_results_df)

# Detailed nested CV results are kept in memory and summarized later.

selected_model_name = nested_results_df.iloc[0]["Model"]

print("\nSELECTED MODEL ACCORDING TO NESTED CV:")
print(selected_model_name)


# ============================================================
# 9. Final hyperparameter tuning on the full training set
# ============================================================

# After the best algorithm has been selected by nested CV,
# we tune this selected algorithm one more time on the full training set.
selected_model_data = models[selected_model_name]
selected_param_grid = selected_model_data["params"] if selected_model_data["params"] else [{}]

final_search = GridSearchCV(
    estimator=selected_model_data["pipeline"],
    param_grid=selected_param_grid,
    scoring=main_scoring,
    cv=inner_cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False
)

final_search.fit(X_train, y_train)

final_model = final_search.best_estimator_

print("\n" + "=" * 80)
print("FINAL MODEL FITTED ON FULL TRAINING SET")
print("=" * 80)

print(f"Selected algorithm: {selected_model_name}")
print("Best parameters on full training set:")
print(final_search.best_params_)
print(f"Best inner CV F1 on full training set: {final_search.best_score_:.4f}")


# ============================================================
# 10. Final evaluation on the held-out test set
# ============================================================

# The test set is used only here.
y_test_pred = final_model.predict(X_test)

final_test_results = {
    "Model": selected_model_name,
    "Best parameters": final_search.best_params_,
    "Test accuracy": accuracy_score(y_test, y_test_pred),
    "Test balanced accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Test precision": precision_score(y_test, y_test_pred, zero_division=0),
    "Test recall": recall_score(y_test, y_test_pred, zero_division=0),
    "Test F1-score": f1_score(y_test, y_test_pred, zero_division=0),
}

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

for metric, value in final_test_results.items():
    print(f"{metric}: {value}")

print("\nClassification report:")
report_text = classification_report(y_test, y_test_pred, zero_division=0)
print(report_text)

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)



# ============================================================
# 11. Graphs and model interpretation
# ============================================================

# 11.1 Model comparison according to outer CV F1-score
# This is the main comparison plot for nested cross-validation.
plot_df = nested_results_df.sort_values("Outer CV F1 mean", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(
    plot_df["Model"],
    plot_df["Outer CV F1 mean"],
    xerr=plot_df["Outer CV F1 std"],
    capsize=4
)
plt.xlabel("Mean outer CV F1-score")
plt.title("Model Comparison by Nested Cross-Validation")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison_outer_cv_f1_std.png", dpi=300)
plt.close()


# 11.2 Confusion matrix on the final held-out test set
# The confusion matrix is computed only from the final test predictions.
final_cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5, 4))
plt.imshow(final_cm)
plt.title(f"Confusion Matrix on Final Test Set - {selected_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["Normal", "Fatigued"])
plt.yticks([0, 1], ["Normal", "Fatigued"])

for i in range(final_cm.shape[0]):
    for j in range(final_cm.shape[1]):
        plt.text(j, i, final_cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# 11.3 Permutation importance on the final held-out test set
# This estimates how much the final test F1-score decreases
# when each feature is randomly shuffled.
perm = permutation_importance(
    final_model,
    X_test,
    y_test,
    scoring=main_scoring,
    n_repeats=30,
    random_state=seed,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Permutation importance mean": perm.importances_mean,
    "Permutation importance std": perm.importances_std
}).sort_values(by="Permutation importance mean", ascending=False)

print("\nPERMUTATION IMPORTANCE ON FINAL TEST SET")
print(perm_df)

perm_plot_df = perm_df.sort_values("Permutation importance mean", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(
    perm_plot_df["Feature"],
    perm_plot_df["Permutation importance mean"],
    xerr=perm_plot_df["Permutation importance std"],
    capsize=3
)
plt.xlabel("Decrease in final test F1-score after permutation")
plt.title(f"Permutation Importance on Final Test Set - {selected_model_name}")
plt.tight_layout()
plt.savefig("final_test_permutation_importance.png", dpi=300)
plt.close()


# 11.4 Logistic Regression coefficients
# These coefficients are learned during final model training.
# They are not computed from the test set.

if selected_model_name == "Logistic Regression":
    logreg_clf = final_model.named_steps["clf"]

    logreg_coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": logreg_clf.coef_[0],
        "Absolute coefficient": np.abs(logreg_clf.coef_[0])
    }).sort_values(by="Absolute coefficient", ascending=False)

    print("\nLOGISTIC REGRESSION COEFFICIENTS OF FINAL MODEL")
    print(logreg_coef_df)

    coef_plot_df = logreg_coef_df.sort_values("Coefficient", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(
        coef_plot_df["Feature"],
        coef_plot_df["Coefficient"]
    )
    plt.xlabel("Coefficient value")
    plt.title("Logistic Regression Coefficients of Final Model")
    plt.tight_layout()
    plt.savefig("final_model_logistic_regression_coefficients.png", dpi=300)
    plt.close()

else:
    print(
        f"\nLogistic Regression coefficients were not computed, "
        f"because the selected model is {selected_model_name}."
    )


# ============================================================
# 12. Compact output summary
# ============================================================

# The CSV has three types of rows:
# 1) Nested CV model comparison summary,
# 2) Outer-fold details with best parameters selected by inner CV,
# 3) Final held-out test results.

summary_rows = []

# 12.1 Nested CV model comparison
for _, row in nested_results_df.iterrows():
    summary_rows.append({
        "Section": "Nested CV model comparison",
        "Model": row["Model"],
        "Outer fold": "",
        "Metric": "Outer CV F1 mean",
        "Value": row["Outer CV F1 mean"],
        "Std": row["Outer CV F1 std"],
        "Best parameters": "",
        "Details": "Mean and standard deviation across outer folds"
    })
    summary_rows.append({
        "Section": "Nested CV model comparison",
        "Model": row["Model"],
        "Outer fold": "",
        "Metric": "Outer CV accuracy mean",
        "Value": row["Outer CV accuracy mean"],
        "Std": row["Outer CV accuracy std"],
        "Best parameters": "",
        "Details": "Mean and standard deviation across outer folds"
    })

# 12.2 Outer-fold details
# These rows show which hyperparameters were selected by inner CV
# in each outer fold and how the selected model performed on the
# corresponding outer validation fold.
for _, row in outer_fold_best_params_df.iterrows():
    summary_rows.append({
        "Section": "Outer fold details",
        "Model": row["Model"],
        "Outer fold": row["Outer fold"],
        "Metric": "Outer fold F1",
        "Value": row["Outer fold F1"],
        "Std": "",
        "Best parameters": str(row["Best parameters"]),
        "Details": f"Inner CV best F1: {row['Inner CV best F1']:.6f}"
    })
    summary_rows.append({
        "Section": "Outer fold details",
        "Model": row["Model"],
        "Outer fold": row["Outer fold"],
        "Metric": "Outer fold accuracy",
        "Value": row["Outer fold accuracy"],
        "Std": "",
        "Best parameters": str(row["Best parameters"]),
        "Details": f"Inner CV best F1: {row['Inner CV best F1']:.6f}"
    })

# 12.3 Final held-out test results
summary_rows.extend([
    {
        "Section": "Final held-out test",
        "Model": selected_model_name,
        "Outer fold": "",
        "Metric": "Accuracy",
        "Value": final_test_results["Test accuracy"],
        "Std": "",
        "Best parameters": str(final_search.best_params_),
        "Details": "Final model evaluated once on held-out test set"
    },
    {
        "Section": "Final held-out test",
        "Model": selected_model_name,
        "Outer fold": "",
        "Metric": "Balanced accuracy",
        "Value": final_test_results["Test balanced accuracy"],
        "Std": "",
        "Best parameters": str(final_search.best_params_),
        "Details": "Final model evaluated once on held-out test set"
    },
    {
        "Section": "Final held-out test",
        "Model": selected_model_name,
        "Outer fold": "",
        "Metric": "Precision",
        "Value": final_test_results["Test precision"],
        "Std": "",
        "Best parameters": str(final_search.best_params_),
        "Details": "Positive class metric"
    },
    {
        "Section": "Final held-out test",
        "Model": selected_model_name,
        "Outer fold": "",
        "Metric": "Recall",
        "Value": final_test_results["Test recall"],
        "Std": "",
        "Best parameters": str(final_search.best_params_),
        "Details": "Positive class metric"
    },
    {
        "Section": "Final held-out test",
        "Model": selected_model_name,
        "Outer fold": "",
        "Metric": "F1-score",
        "Value": final_test_results["Test F1-score"],
        "Std": "",
        "Best parameters": str(final_search.best_params_),
        "Details": "Positive class metric"
    },
])

results_summary_df = pd.DataFrame(summary_rows)
results_summary_df.to_csv("results_summary.csv", index=False)