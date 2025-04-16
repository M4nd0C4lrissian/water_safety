import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from filterpy.kalman import KalmanFilter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from itertools import product
import pickle
import pickle
import os.path
import copy

from kalman_forecast import get_series_by_year
from kalman_linear_approx import linear_regression


class MultiModelEnsembleKalmanFilter():
    
    def __init__(self, model_keys, model_ensemble_dict, per_model_e_num, R, Q, M, N):
        self.M = N*M + N
        self.N = N
        
        self.e_num = per_model_e_num
        
        self.C = None
        self.R = R
        self.Q = Q
        
        self.model_keys = model_keys
        self.ensemble_dict = model_ensemble_dict
        
        # 50 x (18 + 2)
        # This aught to be calculated from some initial val
        self.z = None
        self.z_dict = {}
        self.augment_statespace()
        self.total_ensemble()
        
        # we are going to have a shared mean and shared covariance
        self.mu_z = None
        
        # 2 x 20
        self.H = np.zeros((N, self.M))
        self.H[:, self.M - N : ] = np.eye(N)
        # self.F = F
        self.K = np.zeros((self.M, N))
        
    def total_ensemble(self):
        # (50 * b) x 20 (actually bigger than 20 now)
        self.z = np.empty((0, self.M))
        
        for key in self.model_keys:
            self.z = np.append(self.z, values=self.z_dict[key].copy(), axis=0)
    
    def augment_statespace(self):
        for key in self.model_keys:
            e = self.ensemble_dict[key].copy()
            temp = np.zeros((e.shape[0], self.N))
            
            self.z_dict[key] = np.column_stack((e, temp))
    
    def get_ensemble_num(self):
        return self.z.shape[0]
    
    def predict_augmented_state(self, exogeneous_value_dict):
        
        for key in self.model_keys:
            
            # 50 x (18 + len(observation_features))
            e = self.z_dict[key].copy() 
            augmented_ensemble = e.copy()
            
            # iterate for all 50 ensemble members
            for i in range(self.e_num):
                # evaluate the realization of the predicted parameters with the true exogeneous features
                
                linear_prediction_coefs = e[i][:self.M-self.N].reshape((self.N, int((self.M-self.N) / self.N)))
                observation_predictions =  linear_prediction_coefs @ exogeneous_value_dict[key]
                
                augmented_ensemble[i][self.M - self.N : ] = observation_predictions
                
            noise = np.random.multivariate_normal(mean=np.zeros(self.M - self.N), cov=self.Q, size=self.e_num)
            
            augmented_ensemble[:, : self.M - self.N] += noise
                
            self.z_dict[key] = augmented_ensemble
            ## is this necessary? - I think so
            self.total_ensemble()
    
    ## NOTE: these are the only two methods that should be dealing with global self.z
    # ------------------------------------------------------------------------
    def set_ensemble_mean(self):
        self.mu_z = np.mean(self.z, axis=0)
    
    def set_ensemble_covariance(self):
        
        total_variance = np.zeros((self.M, self.M))
        
        #NOTE: input as 1 x 20 vectors - so our transposes are flipped 
        mean_vector = self.mu_z.reshape(-1,1)
        for i in range(self.get_ensemble_num()):
            
            vector = self.z[i].reshape(-1,1)
            total_variance += vector @ vector.T - mean_vector @ mean_vector.T
            
        self.C = total_variance / self.get_ensemble_num()
    
    # --------------------------------------------------------------------------
    
    # with a constant 1 in exogeneous features
    def predict(self, exogeneous_feature_dict):
        self.predict_augmented_state(exogeneous_feature_dict)
        self.set_ensemble_mean()
        self.set_ensemble_covariance()
        
    def update(self, noisy_y_dict):
        self.kalman_gain()
        self.state_update(noisy_y_dict)
    
    # this seems fine
    def kalman_gain(self):
        self.K = self.C @ self.H.T @ np.linalg.pinv(self.H @ self.C @ self.H.T + self.R)
    
    #TODO - fix this ------------------------------------------------------------
    def state_update(self, noisy_y_dict):
        
        for key in self.model_keys:
            noisy_y = noisy_y_dict[key].copy()
            e = self.z_dict[key].copy() 
        
            for i in range(self.e_num):
                # maintaing the 1 x 20 format - hopefully we don't need to reshape constantly (I think transpose should handle this)
                e[i] = e[i] + (self.K @ (noisy_y[i] - self.H @ e[i].T)).T
            
            self.z_dict[key] = e.copy()
        
        self.total_ensemble()

    def get_ensemble_predicted_state(self, key):
        return np.mean(self.z_dict[key][:, : self.M - self.N], axis=0)
    
    def get_predicted_value(self, key):
        return np.mean(self.z_dict[key][:, self.M - self.N : self.M], axis=0)
    
    def get_ensemble(self, key):
        return self.z_dict[key][:, self.M - self.N : self.M].copy()
    
    # ----------------------------------------------------------------------------
    

#NOTE: Today's weather data, yesterday's ecoli
def per_beach_get_training_set(beach, summer_data, input_features, observation_features, input_scaler = None, ecoli_scaler = None):
    
    X, Y = {}, {}

    for year, df in summer_data.items():
        
        X_train, y_train = [], []
        
        df = df.copy(deep=True)

        # given yesterday's ecoli and today's weather
        df[f'{beach}_eColi_prev'] = df[f'{beach}_eColi'].shift(1)
        df[f'{beach}_eColi_change_prev'] = df[f'{beach}_eColi_change'].shift(1)
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

def beach_point_derivative(data, beach_list):
    
    for beach in beach_list:
    
        data[f'{beach}_eColi_change'] = data[f'{beach}_eColi'].diff()
    
    data.dropna(inplace=True)
    
    return data
       
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

def generate_ensemble_mixed_covariance(sample_states, global_sigma, alpha, e_num, seed=42):
    mu = np.mean(sample_states, axis=0)
    sigma = np.cov(sample_states, rowvar=False)
    
    mixed_sigma = alpha* global_sigma + (1 - alpha) * sigma 
    # sigma += 1e-5 * np.eye(sigma.shape[0])
    
    ensemble = []
    np.random.seed(seed)
    
    for i in range(e_num):
        ensemble.append(np.random.multivariate_normal(mu, mixed_sigma))
    
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


def get_parameter_state_sample(beach_names, observation_features, input_features):
    
    for beach in beach_names:
        ## they're all in one dataframe
        
        # do point derivative for each
        # get_series by year (will contain all beach info per year)
        # collectively remove 2009
        
        # we need to separately get X_dict, Y_dict, X_test, Y_test for each year (likely store in a beach-indexed dictionary)
        # they need separate perturbed Y labels - so the prep is going to be mostly independent for all of them, and the predictions will be independent too
        # state updates use common covariance, but independent perturbed y values
      
        
    # predict_augmented_state is going to need to know which ensembles get which exogeneous features (they will change only with regards to )
        
    #just do it all by varying the ensemble code below
        pass
    
def ensemble_kalman(data, beach_names, per_beach_e_num, r_noise, q_noise, alpha):
    
    observation_features = ['eColi', 'eColi_change']
    # this length no longer outright explains the state dimension
    input_features = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    
    # +1 for constant, +2 for lagged variables
    # N = int(len(observation_features) * len(beach_names))
    N = len(observation_features)
    M = len(input_features) + 3
    
    data = beach_point_derivative(data, beach_names)
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}
    del summer_data[2009]

    beach_test_X = {}
    beach_test_Y = {}
    beach_yearly_regression_coefs = {}
    beach_ensemble_targets = {}
    
    for beach in beach_names:
        
        beach_obs_features = [beach + '_' + feat for feat in observation_features]
        beach_input_features = [feat + '_prev' for feat in beach_obs_features] + input_features
        
        X_dict, Y_dict = per_beach_get_training_set(beach, summer_data, beach_input_features, beach_obs_features)
        
        test_X = X_dict.pop(2024).copy()
        # adding column of ones
        test_X = np.column_stack((np.ones((test_X.shape[0], 1)), test_X))
        test_Y = Y_dict.pop(2024).copy()
        
        # store beach-specific test data for later       
        beach_test_X[beach] = test_X
        beach_test_Y[beach] = test_Y
        
        beach_yearly_regression_coefs[beach] = get_year_order_regression_coeffs(X_dict, Y_dict, f'water_safety\\kalman_forecasting\\regression_weights\\{beach}\\')
        
        beach_ensemble_targets[beach] = perturb_observations(test_Y, np.eye(len(beach_obs_features)) * r_noise, per_beach_e_num)
    
    all_sample_states = np.empty((0, len(observation_features)*M))
    for beach, arr in beach_yearly_regression_coefs.items():
        all_sample_states = np.append(all_sample_states, values = arr, axis=0)
    
    global_covariance = np.cov(all_sample_states, rowvar=False)
    
    beach_init_ensemble = {}
    for beach in beach_names:
        
        initial_ensemble, sigma = generate_ensemble_mixed_covariance(beach_yearly_regression_coefs[beach], global_covariance, alpha, per_beach_e_num)
        beach_init_ensemble[beach] = initial_ensemble
    
    # we need to create combined ensemble - either using global covariance and mean across all estimators for each beach, or beach-specific
    # probably beach-specific, that way doing beach-specific updates and such will work as well, but this may be unnecessary - we could also add global mixing - for now lets
        
    # this should be enough setup, we just have to modify some access routines now - 
    
    # Q = global_covariance * q_noise
    
    Q = np.eye(global_covariance.shape[0]) * q_noise
    R = np.eye(N) * r_noise
    
    mmEnKF = MultiModelEnsembleKalmanFilter(beach_names, beach_init_ensemble, per_beach_e_num, R, Q, M, N)
    
    beach_predictions = {}
    beach_true_labels = {}
    ensemble_predictions = {}
    
    for beach in beach_names:
        beach_predictions[beach] = []
        beach_true_labels[beach] = []
        ensemble_predictions[beach] = []
        
    uncertainty_log = []
    kalman_gain_log = []
    
    ## iterating over the days
    for i in range(beach_test_X[beach_names[0]].shape[0]):
        
        exogeneous_features = {}
        for beach in beach_names:
            exogeneous_features[beach] = beach_test_X[beach][i]
        
        ## exogeneous features should be a dict with a single array for each beach
        mmEnKF.predict(exogeneous_features)
        
        targets = {}
        for beach in beach_names:
            
            beach_predictions[beach].append(mmEnKF.get_predicted_value(beach))
            beach_true_labels[beach].append(beach_test_Y[beach][i])
            ensemble_predictions[beach].append(mmEnKF.get_ensemble(beach))
            
            targets[beach] = beach_ensemble_targets[beach][i]
        
        ## targets should be a dict with a 50 x 2 matrix for each beach
        mmEnKF.update(targets)
        
        uncertainty_log.append(mmEnKF.C)
        kalman_gain_log.append(mmEnKF.K)
    
    return beach_true_labels, beach_predictions, ensemble_predictions, uncertainty_log, kalman_gain_log

def plot_kalman(true_labels, predictions, uncertainty, kalman_gain, title):
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(len(true_labels))
    
    ax1.scatter(x, true_labels, label="True E. coli", color='blue', s=10)
    ax1.scatter(x, predictions, label="Predicted E. coli", color='orange', s=10)
    ax1.axhline(y=100, linestyle='--', color='gray', alpha=0.5)
    # ax1.set_ylim([-10, 300])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("E. coli Level")
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(x, uncertainty, label="Uncertainty Magnitude", color='green', linewidth=1.5)
    ax2.set_ylabel("Prediction Uncertainty (||P||)")
    ax2.legend(loc='upper right')
    
    ax2.plot(x, kalman_gain, label="Kalman Gain", color='red', linestyle='--', alpha=0.7)
    ax2.legend(loc='center right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"water_safety/kalman_forecasting/multi_beach_ensemble/{title}.png")
    plt.close()
    
def visualize_ensemble_spread_vs_truth(beach_names, beach_true_labels, beach_predictions, title_prefix="ensemble_spread_vs_truth"):
    for beach in beach_names:
        true_y = np.array(beach_true_labels[beach])      # Expected shape: (T,) or (T, d)
        pred_y = np.array(beach_predictions[beach])      # Expected shape: (T, d, N)

        # Handle 1D true_y by reshaping to (T, 1)
        if true_y.ndim == 1:
            true_y = true_y.reshape(-1, 1)  # (T, 1)

        # Handle pred_y with shape (T, N) -> convert to (T, 1, N)
        if pred_y.ndim == 2:
            pred_y = pred_y[:, np.newaxis, :]  # (T, 1, N)

        # Calculate norm of true value per timestep (T,)
        mean_truth = np.linalg.norm(true_y, axis=1)

        # Compute standard deviation across ensemble members at each timestep
        spread = np.std(pred_y, axis=2)  # (T, d)

        # Collapse spread into a norm (T,)
        spread_norm = np.linalg.norm(spread, axis=1)

        # Linear regression: fit spread as a function of true value magnitude
        model = LinearRegression()
        model.fit(mean_truth.reshape(-1, 1), spread_norm)
        spread_fit = model.predict(mean_truth.reshape(-1, 1))

        # Plot
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=mean_truth, y=spread_norm, label="Observed Spread", color='green', s=20)
        sns.lineplot(x=mean_truth, y=spread_fit, color='red', label='Linear Fit')
        plt.xlabel("||True E. coli||")
        plt.ylabel("Ensemble Spread (||std across ensemble||)")
        plt.title(f"{title_prefix}_{beach}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"water_safety/kalman_forecasting/multi_beach_ensemble/{title_prefix}_{beach}.png")
        plt.close()


#TODO: grid-searh parameter tuning ()
if __name__ == '__main__':
    
    r_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    q_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    e_list = [5, 10, 15, 25, 50, 75, 100]
    a_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    beach_names = ['HanlansPoint', 'GibraltarPoint', 'CherryBeach', 'WardsIsland', 'CentreIslandBeach']
    
    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_multi_beach.csv', index_col=0)
    
    best_score = {}
    best_params = {}
    best_labels = {}
    best_predictions = {}
    best_uncertainty_log = {}
    best_kalman_gain_log = {}
    
    best_total_score = float('inf')
    best_total_params = None
    
    for beach in beach_names:
        best_score[beach] = float('inf')
    
    for e, a, q, r in product(e_list, a_list, q_list, r_list):
        
        print(f'testing {e}, {a}, {q}, {r}')
        
        ensemble_num = e
        r_noise = r
        q_noise = q
        alpha = a
        
        beach_true_labels, beach_predictions, ensemble_predictions, uncertainty_log, kalman_gain_log = ensemble_kalman(data, beach_names, ensemble_num, r_noise, q_noise, alpha)
        
        uncertainty_mag = [np.linalg.norm(c) for c in uncertainty_log]
        kalman_mag = [np.linalg.norm(k) for k in kalman_gain_log]
        
        total_score = 0
        
        for beach in beach_names:
        
            eColi_true = [t[0] for t in beach_true_labels[beach]]
            eColi_pred = [300 if (t[0] > 300) else t[0] if (t[0] > 0) else 0 for t in beach_predictions[beach]]
        
            mse = mean_squared_error(eColi_true, eColi_pred)
            total_score += mse
            
            if mse < best_score[beach]:
                best_score[beach] = mse
                best_params[beach] = (e, a, q, r)
                best_labels[beach] = eColi_true
                best_predictions[beach] = eColi_pred
                
                best_uncertainty_log[beach] = uncertainty_log
                best_kalman_gain_log[beach] = kalman_gain_log
                
        if total_score < best_total_score:
            best_total_score = total_score
            best_total_params = best_params
    
    for beach in beach_names:
        
        tup = best_params[beach]
        
        mse = best_score[beach]
        
        ensemble_num, alpha, q_noise, r_noise = tup[0], tup[1], tup[2], tup[3]
              
        
        plot_kalman(best_labels[beach], best_predictions[beach], uncertainty_mag, kalman_mag, f'{beach} e_num - {ensemble_num} - a - {alpha} - q - {q_noise} r - {r_noise} - MSE - {mse}')
            
        visualize_ensemble_spread_vs_truth(best_predictions[beach], best_labels[beach], ensemble_predictions)
    
    
    
    
# pretty good params 

#   ensemble_num = 50
#     r_noise = 0.1
#     q_noise = 0.001
    
#     alpha = 0.0001