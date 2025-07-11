import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression
import pickle
import pickle
import os.path
import copy


def plot(true_labels, predictions, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(true_labels)), true_labels, label="True E. coli")
    plt.scatter(range(len(true_labels)), predictions, label="Predicted E. coli", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("E. coli Level")
    plt.legend()
    plt.title(f"{title}")
    plt.savefig("water_safety/kalman_forecasting/graphs//" + title)
    
def get_year(x, dt_format = "%Y-%m-%d"):
    dt = datetime.strptime(x, dt_format)
    return dt.strftime("%Y")

def get_series_by_year(data, year):
    dates = data['CollectionDate']
    
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
        df = df.copy(deep=True)

        df['eColi_prev'] = df['eColi'].shift(1)
        df['eColi_change_prev'] = df['eColi_change'].shift(1)
        df.dropna(inplace=True)
        
        X_train.append(df[features].values)
        y_train.append(df[['eColi', 'eColi_change']].values)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    
    return X_train, y_train

def linear_regression(file_path, X_train, y_train):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        print('Loading existing regressor ----------------------------------')
        
        reg = LinearRegression()
        reg.intercept_ = weights[:,0]
        reg.coef_ = weights[:, 1:]


    else:
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        coeffs = reg.coef_
        intercept = reg.intercept_.reshape(-1,1)

        # Save trained model
        with open(file_path, "wb") as f:
            pickle.dump(np.hstack((intercept, coeffs)), f)

    return reg

def hourly_rollout(kf, hourly_data, control_features):
    
    for i in range(hourly_data.shape[0] - 1):
        kf.predict(u=hourly_data.iloc[i][control_features].values)
        kf.update(kf.x[0:2])
    return kf


# to predict past a single day, this would only work with using forecasted weather updates, which propagates its own error - nevermind.
# TODO change all this - needs to make sure that it only uses the samples from a given day as control
def hourly_kalman(data, q_noise, r_noise, gamma, f_estimate_func=linear_regression):

    data = point_derivative(data)
    
    ## need to consider the day before's ecoli - not the 
    ecoli_features = ['eColi_prev', 'eColi_change_prev']
    
    control_features = ecoli_features + ['Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Precip. Amount (mm)', 'Stn Press (kPa)']
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2014, 2025)}

    test_year = sorted(summer_data.keys())[-1]  # hold out last summer
    test_set = summer_data.pop(test_year).copy()
        # Predict on the held-out summer
    df2 = test_set.copy(deep=True)
    test_set['eColi_prev'] = df2['eColi'].shift(1)
    test_set['eColi_change_prev'] = df2['eColi_change'].shift(1)
    test_set.dropna(inplace=True)

    # TODO: consider these things - or let Oliver's work distinguish if lagged features are useful
    # add lagged features?
    
    # NOTE: only the last hour will have the accurate next day's ecoli level

    X_train, y_train = get_training_set(summer_data, control_features)
    file_path = "linear_transition_coeffs.pkl"
    reg = f_estimate_func(file_path, X_train, y_train)

    B_estimated = np.eye(len(control_features))
    # set first two rows to the linear regression coefficients
    B_estimated[:2, :] = reg.coef_

    #state / control update rate gamma

    # init Kalman Filter
    kf = KalmanFilter(dim_x=len(ecoli_features), dim_z=2)  # dim_z = [eColi, rate of change]
    kf.B = gamma * B_estimated
    kf.F = (1 - gamma) * np.eye(len(ecoli_features), len(ecoli_features))
    
    
    kf.H = np.eye(2, len(ecoli_features))  # Observation matrix: we only observe [eColi, rate of change]
    kf.Q *= q_noise  # Process noise
    kf.R *= r_noise # Measurement noise

    ##update state one step here
    state = test_set.iloc[0][ecoli_features].values
    kf.x = np.array(state).reshape(-1, 1)  # Set initial state
    kf.P *= 1.0  # Reset uncertainty

    predictions = []
    linear_predictions = []
    true_labels = []
    i = 1

    ## test_year
    ## test_set
    days_in_order = test_set['CollectionDate'].unique().values
    
     # forget horizon for a moment
        
    horizon = 1

    for day in days_in_order:
        if i > test_set.shape[0] - horizon:
            break
        
        ## get all rows (hours) for that day
        
        mask = (test_set.loc['CollectionDate'].dt == datetime.strptime(day, "%Y-%m-%d"))
        hourly_data = test_set.iloc[mask.index].copy()

        prev_kf = copy.deepcopy(kf)

        # for _ in range(horizon - 1):
        
        # does not update using the final reading from that day
        kf = hourly_rollout(kf, hourly_data, control_features)
        
        target_row = hourly_data.iloc[-1]
        
        true_labels.append(target_row[['eColi', 'eColi_change']].values[0])
        kf.predict()
        kf.update(target_row[['eColi', 'eColi_change']].values)
        predictions.append(kf.x[0, 0])  # Store predicted eColi

        # 
        linear_predictions.append(reg.predict([[control_features].values])[0,0])

        # reset to previous state here
        kf = prev_kf
        kf.predict()
        kf.update(test_set.iloc[i][['eColi', 'eColi_change']].values)
        i += 1
    
    ## need to fix this - something is broken   
    plot(true_labels, predictions, title=f"Kalman Filter - Summer {test_year} - Horizon {horizon}")
    # plot(test_set.iloc[:test_set.shape[0]- (horizon - 1)], linear_predictions, title=f"Linear Regression - Summer {test_year}")