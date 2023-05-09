'''
1.013 Analysis on imported Tetouan dataset using previous analysis from energy_analysis.py (ML translation of data).

Notes:
Tetouan Population: 380,787 (2014)
Laayoune: 217,732 (2014)
'''

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


def reformat_time(time):
    # Desired DateTime format: 1/1/2017 0:00
    # Current date/time format: 1/1/2017, 12:00 AM
    if time[-2:] == 'AM':
        if time[:2] == '12':
            return '0:00'
        return time[:-3]
    parts = time.split(':')

    if parts[0] == '12':
        parts[0] = 0
    return str(12 + int(parts[0])) + ':' + parts[1][:-3]


def formatting():
    dfs = []

    # Original headers:
    # DateTime,Temperature,Humidity,Wind Speed,general diffuse flows,diffuse flows,Zone 1 Power Consumption,Zone 2  Power Consumption,Zone 3  Power Consumption
    # New headers (before updating):
    # date,time,temp,hum,wind,condition

    for month in range(1, 13):
        filepath = f'C:/Users/haley/1013/1.013_capstone/energy_consumption/monthly_data_reformatted/laayoune_weather_data_month{month}_fixed.csv'
        df = pd.read_csv(filepath)
        df['time'] = df['time'].map(reformat_time)
        df["DateTime"] = df[["date", "time"]].apply(" ".join, axis=1)
        df.drop(['ind', 'date', 'time'], axis=1, inplace=True)
        df.rename(columns={
            'temp': 'Temperature',
            'hum': 'Humidity',
            'wind': 'Wind Speed',
            'condition': 'Condition'
            }, inplace=True)
        dfs.append(df)
    
    df = pd.concat(dfs)
    df.to_csv(f'laayoune_weather_alldata_unscaled.csv')
    return df


def change_units(df):
    # ° F = ( °C × 9/5 ) + 32
    df['Temperature'] = df['Temperature'].map(lambda x: round((x*9/5)+32))
    # mph = kmh/1.609344
    df['Wind Speed'] = df['Wind Speed'].map(lambda x: round(x/1.609344))

    return df


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


def generate_model(df, plot=False):
    #Calculating moving average
    df['total'] = df['Zone 1 Power Consumption'] + df['Zone 2 Power Consumption'] + df['Zone 3 Power Consumption']

    # df['SMA30'] = df['total'].rolling(30).mean()
    # df['SMA15'] = df['total'].rolling(15).mean()

    #defining input and target variable
    X_train = df.loc[:'12-31-2017',['Temperature', 'month', 'dayofyear', 'hour', 'dayofweek', 'Wind Speed', 'Humidity']]
    y_train = df.loc[:'12-31-2017', ['total']]
    X_test = df.loc['10-17-2017':,['Temperature', 'month', 'dayofyear', 'hour', 'dayofweek', 'Wind Speed', 'Humidity']]
    y_test = df.loc['10-17-2017':, ['total']]

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

    if plot:
        # Feature Importance
        fi = pd.DataFrame(data=reg.feature_importances_,
                    index=X_train.columns,
                    columns=['importance'])

        fi.sort_values('importance').plot(kind='barh', title='Feature Importance', color = "#011f4b", figsize=(12,10))
        plt.title('Feature Importance Gradient Boosting Regressor', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #Generating plot
        plt.show()

    return y_test, reg


def predict(reg, df, plot=False):
        # Load data from URL using pandas read_csv method
    df = pd.read_csv('C:/Users/haley/1013/1.013_capstone/Tetouan_City_power_consumption_units.csv')
    df.head() # ?

    # df = change_units(df)
    # df.to_csv('Tetouan_City_power_consumption_units.csv')

    #transforming DateTime column into index - necessary?
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)

    df = create_features(df)
    # print(df.columns)

    _, reg = generate_model(df, plot=True)

    output_csv = 'C:/Users/haley/1013/1.013_capstone/laayoune_weather_alldata_unscaled.csv' 
    df_pred = pd.read_csv(output_csv) 
    df_pred.head()
    df_pred = df_pred.set_index('DateTime')
    df_pred.index = pd.to_datetime(df_pred.index)
    df_pred = create_features(df_pred)

    # df_pred['SMA30'] = df_pred['total'].rolling(30).mean()
    # df_pred['SMA15'] = df_pred['total'].rolling(15).mean()

    cols_when_model_builds = reg.get_booster().feature_names
    #reorder the pandas dataframe
    df_pred = df_pred[cols_when_model_builds]

    interim_results = reg.predict(df_pred)
    results = df_pred.copy()
    results['Prediction'] = interim_results
    # results.to_csv('laayoune_prediction.csv')

    if plot:
        ##Printing predictions on chart to visually assess accuracy
        ax = results[['Prediction']].plot(figsize=(25, 8), color = "#011f4b")
        # y_test['prediction'].plot(ax=ax, style='.', color = "orange")
        plt.legend(['Predictions'])

        plt.title('Laayoune Power Consumption Predictions', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Power Consumption in KW', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.show()
    
    return


def scale_city(df):
    tetouan = 380787 # 2014
    laayoune = 217732 # 2014

    growth_rates = [1.58, 1.81, 1.78, 1.75, 1.72, 1.69, 1.9, 1.86] # 2017 to 2022

    # scale by relation to Tetouan
    df['Prediction'] = df['Prediction'].map(lambda x: x*(laayoune/tetouan))

    population = laayoune
    tetouan_population = tetouan
    for rate in growth_rates:
        population += (population * rate/100)
        tetouan_population += (tetouan_population * rate/100)
    
    print(laayoune, population, tetouan_population)
    df['Prediction'] = df['Prediction'].map(lambda x: x*(population/laayoune))

    df.to_csv('laayoune_prediction_scaled.csv')

def scale_time():
    raise NotImplementedError # this will be something, not sure what yet

def scale_growth():
    raise NotImplementedError # this will be something, not sure what yet


def main():
    # predict()
    
    df = pd.read_csv('C:/Users/haley/1013/1.013_capstone/laayoune_prediction_scaled.csv')
    ##Printing predictions on chart to visually assess accuracy
    ax = df[['Prediction']].plot(figsize=(25, 8), color = "#011f4b")
    # y_test['prediction'].plot(ax=ax, style='.', color = "orange")
    plt.legend(['Predictions'])

    plt.title('Laayoune Power Consumption Predictions (Scaled)', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Power Consumption in KW', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()