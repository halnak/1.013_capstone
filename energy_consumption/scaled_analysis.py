'''
1.013 Analysis on imported Tetouan dataset using previous analysis from energy_analysis.py (ML translation of data).

To Do:
Write function for scaling
Import scraped data
'''

import energy_analysis as ea

# use model.predict


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics


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

def all_in_one_function(df):
    #Calculating moving average
    df['SMA30'] = df['Zone 1 Power Consumption'].rolling(30).mean()
    df['SMA15'] = df['Zone 1 Power Consumption'].rolling(15).mean()

    #defining input and target variable
    X_train = df.loc[:'10-01-2017',['Temperature','dayofyear', 'hour', 'dayofweek', 'SMA30', 'SMA15']]
    y_train = df.loc[:'10-01-2017', ['Zone 1 Power Consumption']]
    X_test = df.loc['10-01-2017':,['Temperature','dayofyear', 'hour', 'dayofweek', 'SMA30', 'SMA15']]
    y_test = df.loc['10-01-2017':, ['Zone 1 Power Consumption']]

    # import xgboost as xgb
    # from sklearn.metrics import mean_squared_error

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


    # Feature Importance
    fi = pd.DataFrame(data=reg.feature_importances_,
                index=X_train.columns,
                columns=['importance'])

    fi.sort_values('importance').plot(kind='barh', title='Feature Importance', color = "#011f4b", figsize=(12,10))
    # plt.title('Feature Importance Gradient Boosting Regressor', fontsize=15)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # #Generating plot
    # plt.show()



    y_test = pd.DataFrame(y_test)
    y_test['prediction'] = reg.predict(X_test)
    df = df.merge(y_test[['prediction']], how='left', left_index=True, right_index=True)
    df.tail()

    return y_test, reg


# import 
##Function to calculate regression metrics, evaluating accuracy
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    mape = (1- metrics.mean_absolute_percentage_error(y_true, y_pred))


    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print('MAPE: ', round(mape,4))


    #Apply function and print results


def scale(scal_fun, df):
    df = df.copy()
    rows, _ = df.shape
    for i in range(rows):
        df.loc[i].at['prediction'] = scal_fun(df.loc[i].at['prediction'])# or something, change label potentially

    return df


def main():
    # Load data from URL using pandas read_csv method
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00616/Tetuan%20City%20power%20consumption.csv')
    df.head()

    #transforming DateTime column into index - necessary?
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)

    df = create_features(df)

    y_test, reg = all_in_one_function(df)

    regression_results(y_test['PowerConsumption_Zone1'], y_test['prediction'])

    df_pred = pd.read_csv('INSERT FINAL CSV')
    df_pred.head()
    df_pred = df_pred.set_index('DateTime')
    df_pred.index = pd.to_datetime(df_pred.index)
    df_pred = create_features(df_pred)

    results = reg.predict(df_pred) # just need to make sure formatted properly, see above what it was fit to
    # Also ensure that model is instead trained on sum of all zones, instead of just Zone 1


if __name__ == "__main__":
    main()