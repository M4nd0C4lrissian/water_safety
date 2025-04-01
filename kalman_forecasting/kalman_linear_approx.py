import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import StandardScaler

def plot(test_set, predictions, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(test_set['Date/Time'], test_set['eColi'], label="True E. coli")
    plt.scatter(test_set['Date/Time'], predictions, label="Predicted E. coli", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("E. coli Level")
    plt.legend()
    plt.title(f"{title}")
    plt.show()

def get_year(x, dt_format = "%Y-%m-%d"):
    dt = datetime.strptime(x, dt_format)
    return dt.strftime("%Y")

def get_series_by_year(data, year):
    dates = data['Date/Time']
    
    converted_dates = dates.map(get_year)
    
    mask = (converted_dates == str(year))
    
    return data.loc[mask]

def point_derivative(data):
    data['eColi_change'] = data['eColi'].diff()
    data.dropna(inplace=True)
    
    return data

def get_training_set(summer_data, features):
    X_train, y_train = [], []
    for year, df in summer_data.items():
        df['eColi_prev'] = df['eColi'].shift(1)
        df['eColi_change_prev'] = df['eColi_change'].shift(1)
        df.dropna(inplace=True)
        
        X_train.append(df[features].values)
        y_train.append(df[['eColi', 'eColi_change']].values)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    
    return X_train, y_train

if __name__ == '__main__':

    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)

    # add rate of change of e.coli
    data = point_derivative(data)
    
    input_features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    state_features = ['eColi_prev', 'eColi_change_prev'] + input_features
    
    # TODO - separate scaling for test and train
    # center
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data[['eColi', 'eColi_change'] + input_features]), columns=['eColi', 'eColi_change'] + input_features)
   
    means = scaler.mean_
    stds = scaler.scale_
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}

    test_year = sorted(summer_data.keys())[-1]  # hold out last summer
    test_set = summer_data.pop(test_year) 

    # TODO: consider these things - or let Oliver's work distinguish if lagged features are useful
    # add lagged features?
    # features
    # center the data? 

    # get training set
    X_train, y_train = get_training_set(summer_data, state_features)

    # Train transition matrix F via Linear Regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    F_estimated = np.eye(len(state_features))
    # set first two rows to the linear regression coefficients
    F_estimated[:2, :] = reg.coef_

    # Save trained model
    with open("kalman_state_transition_model.pkl", "wb") as f:
        pickle.dump(F_estimated, f)

    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=len(state_features), dim_z=2)  # dim_z = [eColi, rate of change]
    kf.F = F_estimated
    kf.H = np.eye(2, len(state_features))  # Observation matrix: we only observe [eColi, rate of change]
    kf.Q *= 0.2  # Process noise
    kf.R *= 1.0 # Measurement noise

    # Predict on the held-out summer
    test_set['eColi_prev'] = test_set['eColi'].shift(1)
    test_set['eColi_change_prev'] = test_set['eColi_change'].shift(1)
    test_set.dropna(inplace=True)

    state = test_set.iloc[0][state_features].values
    kf.x = np.array(state).reshape(-1, 1)  # Set initial state
    kf.P *= 1.0  # Reset uncertainty

    predictions = []
    linear_predictions = []
    # TODO: reset uncertainty in between summers
    for _, row in test_set.iterrows():
        kf.predict()
        kf.update(row[['eColi', 'eColi_change']].values)
        predictions.append(kf.x[0, 0])  # Store predicted eColi
        linear_predictions.append(reg.predict([row[state_features].values])[0,0])
        
        
    plot(test_set, predictions, title="Kalman Filter Predictions for Summer 2024")
    plot(test_set, linear_predictions, title="Linear Regression Predictions for Summer 2024")
