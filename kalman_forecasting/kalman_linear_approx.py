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
import pickle
import os.path
import copy

def plot_from_test_set(test_set, predictions, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(test_set['Date/Time'], test_set['eColi'], label="True E. coli")
    plt.scatter(test_set['Date/Time'], predictions, label="Predicted E. coli", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("E. coli Level")
    plt.legend()
    plt.title(f"{title}")
    plt.savefig("water_safety/kalman_forecasting/graphs//" + title)

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

def kalman(data, horizon, q_noise, r_noise, f_estimate_func=linear_regression):

    data = point_derivative(data)
    
    input_features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    state_features = ['eColi_prev', 'eColi_change_prev'] + input_features
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}

    test_year = sorted(summer_data.keys())[-1]  # hold out last summer
    test_set = summer_data.pop(test_year).copy()
        # Predict on the held-out summer
    df2 = test_set.copy(deep=True)
    test_set['eColi_prev'] = df2['eColi'].shift(1)
    test_set['eColi_change_prev'] = df2['eColi_change'].shift(1)
    test_set.dropna(inplace=True)

    # TODO: consider these things - or let Oliver's work distinguish if lagged features are useful
    # add lagged features?

    X_train, y_train = get_training_set(summer_data, state_features)
    file_path = "linear_transition_coeffs.pkl"
    reg = f_estimate_func(file_path, X_train, y_train)

    F_estimated = np.eye(len(state_features))
    # set first two rows to the linear regression coefficients
    F_estimated[:2, :] = reg.coef_

    # init Kalman Filter
    kf = KalmanFilter(dim_x=len(state_features), dim_z=2)  # dim_z = [eColi, rate of change]
    kf.F = F_estimated
    kf.H = np.eye(2, len(state_features))  # Observation matrix: we only observe [eColi, rate of change]
    kf.Q *= q_noise  # Process noise
    kf.R *= r_noise # Measurement noise

    ##update state one step here
    state = test_set.iloc[0][state_features].values
    kf.x = np.array(state).reshape(-1, 1)  # Set initial state
    kf.P *= 1.0  # Reset uncertainty

    predictions = []
    linear_predictions = []
    true_labels = []
    i = 1
    # TODO: reset uncertainty in between summers

    # TODO: store temporary kf state and reset it to the old one once the rollout is done
    for ind, row in test_set.iloc[1:].iterrows():
        if i > test_set.shape[0] - horizon:
            break

        prev_kf = copy.deepcopy(kf)

        # kf.predict()
        for _ in range(horizon - 1):
            kf.predict()
            kf.update(kf.x[0:2])
        
        true_labels.append(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change']].values[0])
        kf.predict()
        kf.update(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change']].values)
        predictions.append(kf.x[0, 0])  # Store predicted eColi

        # linear still predicts only next day
        linear_predictions.append(reg.predict([row[state_features].values])[0,0])

        # reset to previous state here
        kf = prev_kf
        kf.predict()
        kf.update(test_set.iloc[i][['eColi', 'eColi_change']].values)
        i += 1
    
    ## need to fix this - something is broken   
    plot(true_labels, predictions, title=f"Kalman Filter - Summer {test_year} - Horizon {horizon}")
    # plot(test_set.iloc[:test_set.shape[0]- (horizon - 1)], linear_predictions, title=f"Linear Regression - Summer {test_year}")

if __name__ == '__main__':

    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)
    horizon = 1
    kalman(data, horizon, 0.2, 1)
