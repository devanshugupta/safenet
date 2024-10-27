import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, false_positive_rate
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def falsePositiveBarchart(df, group, when):
    df = df.reset_index()
    col = np.array(df[group])
    print(col)
    fp = np.array(df['false_positive_rate'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=col, y=fp, palette="viridis")
    plt.title(f'False Positives by {group}')
    plt.xlabel(f'{group}')
    plt.ylabel('Count of False Positives')
    plt.savefig(f'{when}_{group}.png')
    plt.show()

def detect(df, biased_column, target, when='before'):
    print('For col: ', biased_column)
    y = df[target]
    X = df.drop(columns=[target], axis =1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    sensitive_features_train = df.loc[X_train.index, biased_column]
    # Create MetricFrame to assess fairness

    metric_frame = MetricFrame(metrics={'accuracy': accuracy_score,
                                         'false_positive_rate': false_positive_rate},
                               y_true=y_test,
                               y_pred=y_pred,
                               sensitive_features=df.loc[X_test.index, biased_column])

    if when=='before':
        falsePositiveBarchart(metric_frame.by_group, biased_column, when)

    # Apply Equalized Odds to mitigate bias
    mitigator = ExponentiatedGradient(
        estimator=model,
        constraints=EqualizedOdds()
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    y_pred_mitigated = mitigator.predict(X_test)

    # Check the metrics after mitigation
    metric_frame_mitigated = MetricFrame(metrics={'accuracy': accuracy_score,
                                                  'false_positive_rate': false_positive_rate},
                                          y_true=y_test,
                                          y_pred=y_pred_mitigated,
                                          sensitive_features=df.loc[X_test.index, biased_column])

    if when=='after':
        falsePositiveBarchart(metric_frame_mitigated.by_group, biased_column, when)

    return metric_frame_mitigated.by_group
