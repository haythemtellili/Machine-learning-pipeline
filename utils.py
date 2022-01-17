import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


def dummify_dataset(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df = df.drop([column], axis=1)
    return df


# Evaluation Metrics
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse_score(y, y_pred):
    score = rmse(y, y_pred)
    return score


# Cross-validation RMSLE score
def rmsle_cv(model, X_train, y_train):
    kf = KFold(n_splits=3, shuffle=True,
               random_state=42).get_n_splits(X_train.values)
    # Evaluate a score by cross-validation
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train.values,
            y_train, scoring="neg_mean_squared_error", cv=kf
        )
    )
    return rmse


def rmse_cv_score(model, X_train, y_train):
    score = rmsle_cv(model, X_train, y_train)
    return score


# Feature Importance
def model_feature_importance(model, X_train, model_artifacts_dir):
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=["Importance"],
    )

    # sort by importance
    feature_importance.sort_values(by="Importance",
                                   ascending=False, inplace=True)

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_importance.reset_index(),
        y="index",
        x="Importance",
    ).set_title("Feature Importance")
    # save image
    plt.savefig(f"{model_artifacts_dir}/feature_importance.png",
                bbox_inches="tight")
