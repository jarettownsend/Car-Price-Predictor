import pandas as pd
import numpy as np

def preprocess_data(inputs, model_package):
    """Processes all the input data into what the model expects"""
    df = pd.DataFrame([inputs])
    df['cylinders'] = df['car_name'].map(model_package['car_cylinders_mapping'])
    df['cylinders'] = df['cylinders'].str.extract('(\d+)').astype(int)
    df['condition'] = df['condition'].map(model_package['condition_mapping'])
    df['clean_title'] = df['title_status'].apply(lambda x: 1 if x == 'clean' else 0)
    df = df.drop(columns=['title_status'])
    df['age'] = model_package['reference_year'] - df['year']
    df = df.drop(columns=['year'])
    df = pd.get_dummies(df, columns=model_package['encoded_columns'], drop_first=True).astype(int)
    missing_cols = set(model_package['feature_names']) - set(df.columns)
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_cols))
        df = pd.concat([df, missing_df], axis=1)
    df = df[model_package['feature_names']]

    return df

def predict_prices(processed_data, model_package):
    """Predicts the price of a car using the model package"""
    tree_predictions = np.array([tree.predict(processed_data.values)[0] for tree in model_package['model'].estimators_])
    point_estimate = np.mean(tree_predictions)
    std_error = np.std(tree_predictions) / np.sqrt(len(tree_predictions))
    lower_bound = point_estimate - 1.96 * std_error
    upper_bound = point_estimate + 1.96 * std_error

    return point_estimate, lower_bound, upper_bound