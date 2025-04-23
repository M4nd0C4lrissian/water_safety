import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import product
import pickle
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
    plt.axhline(y=100, linestyle='--')
    plt.savefig("water_safety/kalman_forecasting/tuning_graphs//" + title + '.png')

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

def get_training_set(summer_data, features, input_scaler, ecoli_scaler):
    X_train, y_train = [], []
    for year, df in summer_data.items():
        df = df.copy(deep=True)

        df['eColi_prev'] = df['eColi'].shift(1)
        df['eColi_change_prev'] = df['eColi_change'].shift(1)
        df.dropna(inplace=True)
        
        X_train.append(df[features].values)
        y_train.append(df[['eColi', 'eColi_change']].values)

    X = np.vstack(X_train)
    Y = np.vstack(y_train)
    
    if input_scaler:
        X = input_scaler.fit_transform(X)
        
    if ecoli_scaler:
        Y = ecoli_scaler.fit_transform(Y)
        
    
    return X, Y

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

def evaluate_kalman_filter(test_set, kf, F_estimated, state_features, horizon):
    kf.F = F_estimated
    # kf.F =  np.eye(len(state_features))
    kf.H = np.eye(8, len(state_features))  # Observation matrix: we only observe [eColi, rate of change]
    
    input_features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
                    'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    state_features = ['eColi_prev', 'eColi_change_prev'] + input_features

    ##update state one step here
    state = test_set.iloc[0][state_features].values
    kf.x = np.array(state).reshape(-1, 1)  # Set initial state
    kf.P *= 1.0 
    predictions = []
    true_labels = []
    i = 1

    for ind, row in test_set.iloc[1:].iterrows():
        if i > test_set.shape[0] - horizon:
            break

        prev_kf = copy.deepcopy(kf)

        for _ in range(horizon - 1):
            kf.predict()
            # kf.update(kf.x[0:2])
        
        true_labels.append(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change']].values[0])
        kf.predict()
        predictions.append(kf.x[0, 0])  # Store predicted eColi
        ## NOTE: these values are not scaled - and they should be for this comparison
        kf.update(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change'] + input_features].values)

        # reset to previous state here - to keep the rollout update or not?
        # NOTE: when I update according to 
        kf = prev_kf
        kf.predict()
        ## NOTE: these values are not scaled - and they should be for this comparison
        kf.update(test_set.iloc[i][['eColi', 'eColi_change'] + input_features].values)
        # print(kf.K)
        i += 1
        
    return mean_squared_error(true_labels, predictions), true_labels, predictions


def run_optimization_pipeline(data, horizon, q_vals, r_vals, alpha_vals, input_scaler, ecoli_scaler, num_test_years):
    data = point_derivative(data)

    input_features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
                      'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    state_features = ['eColi_prev', 'eColi_change_prev'] + input_features

    # Get summers by year
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}
    
    test_years = []
    test_set = pd.DataFrame(columns=['eColi','statusFlag','Date/Time','Max Temp (°C)','Min Temp (°C)','Mean Temp (°C)','Heat Deg Days (°C)','Cool Deg Days (°C)','Total Precip (mm)'])
    for i in range(-num_test_years, 0):
        test_year = sorted(summer_data.keys())[i]
        test_years.append(test_year)
        new = summer_data.pop(test_year).copy()
        test_set = pd.concat([test_set, new], ignore_index=True)
        
    test_set = test_set.reindex(range(test_set.shape[0]))

    X_train, y_train = get_training_set(summer_data, state_features, input_scaler, ecoli_scaler)

    df2 = test_set.copy()
    test_set['eColi_prev'] = df2['eColi'].shift(1)
    test_set['eColi_change_prev'] = df2['eColi_change'].shift(1)
    test_set.dropna(inplace=True)
    
    if input_scaler:
        test_set[state_features] = input_scaler.transform(test_set[state_features])
    
    if ecoli_scaler:
        test_set[['eColi', 'eColi_change']] = ecoli_scaler.transform(test_set[['eColi', 'eColi_change']])

    best_score = float('inf')
    best_params = None
    best_labels = None
    best_predictions = None

    for alpha, q, r in product(alpha_vals, q_vals, r_vals):
        print(f'Parameters: a - {alpha}, Q - {q}, R - {r}')
        
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)

        F_est = np.eye(len(state_features))
        F_est[:2, :] = reg.coef_
        
        B = np.eye(len(state_features))
        B[:2, :] = np.zeros((2, len(state_features)))
        

        kf = KalmanFilter(dim_x=len(state_features), dim_z=8)
        kf.Q *= q
        kf.R *= r
        # kf.B = B

        try:
            mse, true_labels, predictions = evaluate_kalman_filter(test_set, kf, F_est, state_features, horizon)
            if mse < best_score:
                best_score = mse
                best_params = (alpha, q, r)
                best_labels = true_labels
                best_predictions = predictions
        except Exception as e:
            print(f"Skipping combo alpha={alpha}, q={q}, r={r}: {e}")
            continue

    print(f"\n Best Params: alpha={best_params[0]}, Q={best_params[1]}, R={best_params[2]} with MSE={best_score:.4f}")
    # print(best_predictions)
    plot(best_labels, best_predictions, title=f"Weather Features Kalman Filter alpha={best_params[0]}, Q={best_params[1]}, R={best_params[2]} - Summer {test_years} - Horizon {horizon}")
    return best_params
    

if __name__ == '__main__':

    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)
    horizon = 5
    
    input_scaler = None
    ecoli_scaler = None
    best_params = run_optimization_pipeline(data, horizon, num_test_years=1, input_scaler=input_scaler, ecoli_scaler=ecoli_scaler, q_vals=[1e-5,1e-4,1e-3, 0.01], r_vals=[1e-5,1e-4,1e-3, 0.01, 1, 10], alpha_vals= [1.0, 10.0])

# chooses q really low and R really high to minimize MSE by flattening the predictions
#------------------------------------------------------------------------------------------------------------------------

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

    X_train, y_train = get_training_set(summer_data, state_features, None, None)
    file_path = "linear_transition_coeffs.pkl"
    reg = f_estimate_func(file_path, X_train, y_train)

    ## biggest assumption here - that weather conditions do not change as we rollout our state predictions (we do not try to model these)
    F_estimated = np.eye(len(state_features))
    # set first two rows to the linear regression coefficients
    F_estimated[:2, :] = reg.coef_

    # init Kalman Filter
    kf = KalmanFilter(dim_x=len(state_features), dim_z=8)  # dim_z = [eColi, rate of change]
    kf.F = F_estimated
    # kf.F =  np.eye(len(state_features))
    kf.H = np.eye(8, len(state_features))  # Observation matrix: we only observe [eColi, rate of change]
    kf.Q *= q_noise  # Process noise
    kf.R *= r_noise # Measurement noise
    
    B = np.eye(len(state_features))
    B[:2, :] = np.zeros((2, len(state_features)))
    
    F_control = np.zeros((len(state_features), len(state_features)))
    F_control[:2, :] = reg.coef_

    ##update state one step here
    state = test_set.iloc[0][state_features].values
    kf.x = np.array(state).reshape(-1, 1)  # Set initial state
    kf.P *= 1.0 
    predictions = []
    linear_predictions = []
    true_labels = []
    i = 1

    for ind, row in test_set.iloc[1:].iterrows():
        if i > test_set.shape[0] - horizon:
            break

        prev_kf = copy.deepcopy(kf)

        # kf.predict()
        for _ in range(horizon - 1):
            kf.K = F_estimated
            kf.predict()
            # kf.update(kf.x[0:2])
        
        true_labels.append(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change']].values[0])
        # kf.predict(u = test_set.iloc[i][state_features].values.reshape(-1, 1))
        kf.predict()
        predictions.append(kf.x[0, 0])  # Store predicted eColi
        kf.update(test_set.iloc[i + horizon - 1][['eColi', 'eColi_change'] + input_features].values)

        # linear still predicts only next day
        linear_predictions.append(reg.predict([row[state_features].values])[0,0])

        # reset to previous state here - to keep the rollout update or not?
        kf = prev_kf
        kf.K = F_estimated
        # kf.predict(u = test_set.iloc[i][state_features].values.reshape(-1, 1))
        kf.predict()
        kf.update(test_set.iloc[i][['eColi', 'eColi_change'] + input_features].values)
        # print(kf.K)
        i += 1
    
    ## need to fix this - something is broken   
    plot(true_labels, predictions, title=f"{horizon}-step Forecasting")
    # plot(true_labels, linear_predictions, title=f"Linear Regression - Summer {test_year}")
    
    
# if __name__ == '__main__':

#     data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)
#     horizon = 5
#     kalman(data, horizon, 0.2, 1)