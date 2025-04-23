import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from filterpy.kalman import KalmanFilter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import product
import pickle
import pickle
import os.path
import copy

from kalman_forecast import plot_kalman, point_derivative, get_series_by_year
from old_inaccurate_kalman_linear_approx import linear_regression


class EnsembleKalmanFilter():
    
    def __init__(self, e_0, R, Q, M, N):
        self.M = N*M + N
        self.N = N
        
        self.C = None
        self.R = R
        self.Q = Q
        
        # 50 x (18 + 2)
        temp = np.zeros((e_0.shape[0], N))
        # This aught to be calculated from some initial val
        self.z = np.column_stack((e_0, temp))
        self.mu_z = None
        
        # 2 x 20
        self.H = np.zeros((N, self.M))
        self.H[:, self.M - N : ] = np.eye(N)
        # self.F = F
        self.K = np.zeros((self.M, N))
        
    def get_ensemble_num(self):
        return self.z.shape[0]
    
    #TODO: add constant 1 to weather data -> 1 x 9 (added constant 1 value) - need to remember to do that
    def predict_augmented_state(self, exogeneous_values):
        
        # 50 x (18 + len(observation_features)) 
        augmented_ensemble = self.z.copy()
        
        # iterate for all 50 ensemble members
        for i in range(self.get_ensemble_num()):
            # evaluate the realization of the predicted parameters with the true exogeneous features
            
            linear_prediction_coefs = self.z[i][:self.M-self.N].reshape((self.N, int((self.M-self.N) / self.N)))
            observation_predictions =  linear_prediction_coefs @ exogeneous_values
            
            augmented_ensemble[i][self.M - self.N : ] = observation_predictions
            
        noise = np.random.multivariate_normal(mean=np.zeros(self.M - self.N), cov=self.Q, size=self.get_ensemble_num())
        
        augmented_ensemble[:, : self.M - self.N] += noise
            
        self.z = augmented_ensemble
        
    def set_ensemble_mean(self):
        self.mu_z = np.mean(self.z, axis=0)
    
    def set_ensemble_covariance(self):
        
        total_variance = np.zeros((self.M, self.M))
        
        #NOTE: input as 1 x 20 vectors - so our transposes are flipped 
        for i in range(self.get_ensemble_num()):
            
            vector = self.z[i].reshape(-1,1)
            mean_vector = self.mu_z.reshape(-1,1)
            total_variance += vector @ vector.T - mean_vector @ mean_vector.T
            
        self.C = total_variance / self.get_ensemble_num()
            
    # with a constant 1 in exogeneous features
    def predict(self, exogeneous_features):
        self.predict_augmented_state(exogeneous_features)
        self.set_ensemble_mean()
        self.set_ensemble_covariance()
        
    def update(self, noisy_y):
        self.kalman_gain()
        self.state_update(noisy_y)
        
    def kalman_gain(self):
        self.K = self.C @ self.H.T @ np.linalg.pinv(self.H @ self.C @ self.H.T + self.R)
    
    def state_update(self, noisy_y):
        
        for i in range(self.get_ensemble_num()):
            # maintaing the 1 x 20 format - hopefully we don't need to reshape constantly (I think transpose should handle this)
            self.z[i] = self.z[i] + (self.K @ (noisy_y[i] - self.H @ self.z[i].T)).T

    def get_ensemble_predicted_state(self):
        return np.mean(self.z[:, : self.M - self.N], axis=0)
    
    def get_predicted_value(self):
        return np.mean(self.z[:, self.M - self.N : self.M], axis=0)

#NOTE: when integrating multiple beaches - I think we have overlap in the days off, but just in case we can merge and then check gaps - worst case we fill missing days with averages 

#NOTE: Today's weather data, yesterday's ecoli
def get_training_set(summer_data, input_features, observation_features, input_scaler = None, ecoli_scaler = None):
    
    X, Y = {}, {}

    for year, df in summer_data.items():
        
        X_train, y_train = [], []
        
        df = df.copy(deep=True)

        # given yesterday's ecoli and today's weather
        df['eColi_prev'] = df['eColi'].shift(1)
        df['eColi_change_prev'] = df['eColi_change'].shift(1)
        df.dropna(inplace=True)
        
        # predict today's e.Coli
        X_train.append(df[input_features].values)
        y_train.append(df[observation_features].values)

        X_train = np.array(X_train)
        Y_train = np.array(y_train)
        
        if input_scaler:
            X_train = input_scaler.fit_transform(X_train)
            
        if ecoli_scaler:
            Y_train = ecoli_scaler.fit_transform(Y_train)
            
        X[year] = X_train[0]
        Y[year] = Y_train[0]
        
    return X, Y


# remove 2009 Hanlan's        
def fit_regressors_per_year(X_dict, Y_dict, file_path):
    
    regressors = []
    
    for year in np.sort(list(X_dict.keys())):
        regressors.append(linear_regression(file_path + f'{year}.pkl', X_dict[year], Y_dict[year]))
        
    return regressors

def get_year_order_regression_coeffs(X_dict, Y_dict, file_path):
    regressors = fit_regressors_per_year(X_dict, Y_dict, file_path)
    
    coefficients_with_intercept = np.array([np.concatenate((reg.intercept_.reshape(-1,1), reg.coef_), axis=1) for reg in regressors])
    
    flattened_coefficients = np.zeros((coefficients_with_intercept.shape[0], coefficients_with_intercept.shape[1] * coefficients_with_intercept.shape[2]), dtype=np.float32)
    for i in range(coefficients_with_intercept.shape[0]):
        row = coefficients_with_intercept[i]
        flattened_coefficients[i] = row.flatten()
        
    return flattened_coefficients
        
#NOTE: need to change things - have the function that integrates in the evaluation of the parameters G(u) be part of the augmented state, and just have H be a simple selection process
# this way the uncertainty of the model and observations are decoupled

def generate_ensemble(sample_states, e_num, seed=42):
    mu = np.mean(sample_states, axis=0)
    sigma = np.cov(sample_states, rowvar=False)
    
    # sigma += 1e-5 * np.eye(sigma.shape[0])
    
    ensemble = []
    np.random.seed(seed)
    
    for i in range(e_num):
        ensemble.append(np.random.multivariate_normal(mu, sigma))
    
    return np.array(ensemble), sigma

def perturb_observations(true_y, R_matrix, e_num, seed=42):
    
    mu = np.zeros(true_y[0].shape)

    perturbed_y = []
    np.random.seed(seed)
    
    for j in range(true_y.shape[0]):
        ensemble_values = []
        for i in range(e_num):
            ensemble_values.append(true_y[j] + np.random.multivariate_normal(mu, R_matrix)) ## according to paper "Ensemble Kalman Methods for Inverse Problems"
        perturbed_y.append(np.array(ensemble_values))    
        
    return np.array(perturbed_y)

def ensemble_kalman(data, beach, e_num, r_noise, q_noise):
    
    # have to flatten state (coefficients) and make this work for multiple observation features
    observation_features = ['eColi', 'eColi_change']
    input_features = ['eColi_prev', 'eColi_change_prev'] + ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    
    # + 1 for constant 1
    M = len(input_features) + 1
    N = len(observation_features)
    
    data = point_derivative(data)
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}
    
    # Hanlan's 2009 missing a lot of dates
    del summer_data[2009]
    
    X_dict, Y_dict = get_training_set(summer_data, input_features, observation_features)
    
    test_X = X_dict.pop(2024).copy()
    # adding column of ones
    test_X = np.column_stack((np.ones((test_X.shape[0], 1)), test_X))
    test_Y = Y_dict.pop(2024).copy()
    
    parameter_state_sample = get_year_order_regression_coeffs(X_dict, Y_dict, f'water_safety\\kalman_forecasting\\regression_weights\\{beach}\\')
    
    initial_ensemble, sigma = generate_ensemble(parameter_state_sample, e_num)
    
    Q = sigma * q_noise
    R = np.eye(N) * r_noise
    
    ensemble_targets = perturb_observations(test_Y, R, e_num)
   
    ## build EnKF object, 
    ## iterate through test data
    ## predict, update
    ## get_ensemble_predicted_state
    ## unflatten it, and append the prediction / calculate our true MSE
    
    predictions = []
    true_labels = []
    uncertainty_log = []
    kalman_gain_log = []
    
    EnKF = EnsembleKalmanFilter(initial_ensemble, R, Q, M, N)
    
    for i in range(test_X.shape[0]):
        
        exogeneous_features = test_X[i]
        
        EnKF.predict(exogeneous_features)
        
        predictions.append(EnKF.get_predicted_value())
        true_labels.append(test_Y[i])
        
        EnKF.update(ensemble_targets[i])
        uncertainty_log.append(EnKF.C)
        kalman_gain_log.append(EnKF.K)
    
    
    return true_labels, predictions, uncertainty_log, kalman_gain_log

## womp womp - we converge onto the linear estimate
if __name__ == '__main__':
    
    ensemble_num = 50
    r_noise = 50
    q_noise = 0.01
    
    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)
    true_labels, predictions, uncertainty_log, kalman_gain_log = ensemble_kalman(data, 'HanlansPoint', ensemble_num, r_noise, q_noise)
    
    eColi_true = [t[0] for t in true_labels]
    eColi_pred = [t[0] for t in predictions]
    
    uncertainty_mag = [np.linalg.norm(c) for c in uncertainty_log]
    kalman_mag = [np.linalg.norm(k) for k in kalman_gain_log]
    
    mse = mean_squared_error(eColi_true, eColi_pred)
    
    plot_kalman(eColi_true, eColi_pred, uncertainty_mag, kalman_mag, f'Ensemble of size {ensemble_num} q - {q_noise} r - {r_noise} - MSE - {mse}')
    
    # Should I try the new parameters on a final validation set? - should tune on 2023,
    # then predict on 2024 (test how this does)
    # yes, also, proceed as expected - use multiple beaches, see the effects
    # get some final results relative to linear regression and basic Kalman filter