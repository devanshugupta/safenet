
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import os

folder_path = 'plots'
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
'''def visualize_column_correlation(dataFrame, targetColumn, arrayOfBiasColumns):
    updatedArrayOfBiasColumns = []
    for BiasColumn in arrayOfBiasColumns:
        updatedArrayOfBiasColumns = []

    for column in arrayOfBiasColumns:
        # Step 1: Distribution analysis with respect to target column
        distribution_df = dataFrame[column].value_counts().reset_index()
        distribution_df.columns = [column, 'Count']

        # Create a DataFrame suitable for plotting
        distribution_df.plot(kind='bar', x=column, y='Count', stacked=True, figsize=(12, 8))

        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.savefig(f'plots/distribution_{column}.png')
        plt.clf()

        # Step 2: Check for imbalance (e.g., check if one class dominates)
        imbalance_check = dataFrame[column].value_counts(normalize=True)
        if any(imbalance_check > 0.7):
            print(f"Imbalance detected in column: {column}")

        # Step 3: Bias detection using correlation
        dataFrame = dataFrame.select_dtypes(include='number')
        corr_matrix = dataFrame.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title(f'Correlation Heatmap for {column} and {targetColumn}')
        plt.savefig(f'plots/correlation_heatmap_{column}.png')
        plt.clf()

        # Check if correlation value is high (you can adjust threshold based on your requirement)
        if abs(corr_matrix.at[column, targetColumn]) > 0.5:  # example threshold
            print(f"High correlation detected in column: {column}")

        # Step 4: Chi-Squared Test for independence
        contingency_table = pd.crosstab(dataFrame[column], dataFrame[targetColumn])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value < 0.05:
            print(f"Chi-Squared Test indicates dependence in column: {column}")
            updatedArrayOfBiasColumns.append(column)

    return updatedArrayOfBiasColumns

'''
def plot_bias(df, biased_columns, target):
    # Set the visual style of seaborn
    sns.set(style="whitegrid")

    for column in biased_columns:
        # Check if the column exists in the DataFrame
        if column not in df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            continue

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Count plot for categorical biased column
        sns.countplot(data=df, x=column, hue=target, ax=axes[0])
        axes[0].set_title(f'Distribution of {target} by {column}')
        axes[0].set_ylabel('Count')
        axes[0].set_xlabel(column)

        # Box plot for numerical target variable against biased column
        if df[target].dtype in ['int64', 'float64']:  # Check if target is numeric
            sns.boxplot(data=df, x=column, y=target, ax=axes[1])
            axes[1].set_title(f'{target} Distribution by {column}')
            axes[1].set_ylabel(target)
            axes[1].set_xlabel(column)

        plt.tight_layout()
        plt.savefig(f'plots/{column}_bias_visualization.png')
        plt.close(fig)


