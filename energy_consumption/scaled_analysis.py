'''
1.013 Analysis on imported Tetouan dataset using previous analysis from energy_analysis.py (ML translation of data).

To Do:
Write function for scaling based on report
Import scraped data
'''

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


def formatting():
    # Just formatting the .csv of scraped data into the same format/headers as the original dataset
    raise Exception("Not implemented yet")


def check_missed(df):
    #Checking % of Null values within dataset
    missed = pd.DataFrame()
    missed['column'] = df.columns
    missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]
    missed = missed.sort_values('percent',ascending=False)

    return missed

    
## Feature Engineering extracts the hour, day of the week, quarter, month etc. from the datetime index
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def generate_model(df):
    #Calculating moving average
    df['total'] = df['Zone 1 Power Consumption'] + df['Zone 2 Power Consumption'] + df['Zone 3 Power Consumption']

    df['SMA30'] = df['total'].rolling(30).mean()
    df['SMA15'] = df['total'].rolling(15).mean()

    #defining input and target variable
    X_train = df.loc[:'10-01-2017',['Temperature','dayofyear', 'hour', 'dayofweek', 'SMA30', 'SMA15']]
    y_train = df.loc[:'10-01-2017', ['total']]
    X_test = df.loc['10-01-2017':,['Temperature','dayofyear', 'hour', 'dayofweek', 'SMA30', 'SMA15']]
    y_test = df.loc['10-01-2017':, ['total']]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                        n_estimators=1500,
                        early_stopping_rounds=50,
                        objective='reg:linear',
                        max_depth=6,
                        learning_rate=0.03, 
                        random_state = 48)
    reg.fit(X_train, y_train,         
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_test = pd.DataFrame(y_test)
    y_test['prediction'] = reg.predict(X_test)
    df = df.merge(y_test[['prediction']], how='left', left_index=True, right_index=True)
    df.tail()

    return y_test, reg


def predict(reg, df):
    return reg.predict(df)


def scale(scal_fun, df):
    df = df.copy() # do not modify original input
    rows, _ = df.shape
    for i in range(rows):
        df.at[i, 'prediction'] = scal_fun(df.at[i, 'prediction']) # may change label

    return df


def main():
    # Load data from URL using pandas read_csv method
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00616/Tetuan%20City%20power%20consumption.csv')
    df.head()

    #transforming DateTime column into index - necessary?
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)

    df = create_features(df)

    _, reg = generate_model(df)

    output_csv = '' # TODO: Fill in with final output .csv, once it's created
    df_pred = pd.read_csv(output_csv) 
    df_pred.head()
    df_pred = df_pred.set_index('DateTime')
    df_pred.index = pd.to_datetime(df_pred.index)
    df_pred = create_features(df_pred)

    results = scale(predict(reg, df_pred)) 


if __name__ == "__main__":
    main()