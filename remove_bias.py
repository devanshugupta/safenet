# IMPORTS

import pandas as pd
from data_cleaning import clean_data
from get_bias import Generator
from stats import plot_bias
from imblearn.over_sampling import SMOTE, ADASYN
from detect_bias import detect
from flask import Flask, request, jsonify
from flask_cors import CORS


import warnings
warnings.filterwarnings("ignore")


def transform(data, target):
    if not target:
        target = data.columns[-1]
    generator = Generator(data, target)
    result = generator.call_openai_api(
                engine= 'gpt-4o-mini',
                max_tokens=512,
                temperature=0.2,
                top_p=0.2,
                n=1,
                stop='\n\n',
                is_chat=True
            )
    
    result = result['choices'][0]['message']['content']
    print(result)
    
    lines = result.split('\n')

    df, y = clean_data(data, target)
    df = pd.concat([df, y], axis=1)
    print(df.describe())
    biased_columns = []
    for line in lines:
        name = line.split(':')[0].replace('- ', '').strip()
        if name in df.columns:
            biased_columns.append(name)

    final = []
    for name in biased_columns:
        if len(df[name].unique()) <= 20:
            final.append(name)

    biased_columns = final
    print('Columns predicted to have bias: ',biased_columns)

    for biased_column in biased_columns:
        bias_by_group = bias_by_group = detect(df, biased_column, target)
        print(bias_by_group)
    #plot_bias(df, [biased_columns[0]], target)
    def mitigate_bias(df, target, biased_columns):
        """
        Detects and mitigates bias in the specified columns using AIF360, Fairlearn, or SMOTE.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            target (str): The name of the target column.
            biased_columns (list): List of column names that might be biased.
            mitigation_method (str): The mitigation method to use. Options are "AIF360", "Fairlearn", "SMOTE".

        Returns:
            pd.DataFrame: A new DataFrame with bias mitigated based on the chosen method.
        """
        y = df[target]
        X = df.drop([target], axis=1)
        try:
            smote = ADASYN(random_state=42, sampling_strategy='minority')
            X,y = smote.fit_resample(X,y)
        except:
            try:
                smote = SMOTE(random_state=42, sampling_strategy='minority')
                X, y = smote.fit_resample(X, y)
            except:
                return df
        resampled_df = pd.concat([X,y], axis=1)
        return resampled_df

    for col in df.columns:
        df = df[df[col].notna()]
    mitigated_df = mitigate_bias(df, target, biased_columns)

    print('After mitigation : ')
    try:
        for biased_column in biased_columns:
            bias_by_group = detect(mitigated_df, biased_column, target, 'after')
            print(bias_by_group)

    except:
        print('... Not detecting for multi-class')

data = pd.read_csv('adult.csv')
transform(data, '')
