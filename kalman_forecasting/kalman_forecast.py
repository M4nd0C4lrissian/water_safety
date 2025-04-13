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

#try to add better logging to this one
# should probably plot gaps?

def plot_kalman(true_labels, predictions, uncertainty, kalman_gain, title):
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(len(true_labels))
    
    ax1.scatter(x, true_labels, label="True E. coli", color='blue', s=10)
    ax1.scatter(x, predictions, label="Predicted E. coli", color='orange', s=10)
    ax1.axhline(y=100, linestyle='--', color='gray', alpha=0.5)
    ax1.set_ylim([-10, 300])
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
    plt.savefig(f"water_safety/kalman_forecasting/forecast_graphs/{title}.png")
    plt.close()
    
def get_year(x, dt_format = "%Y-%m-%d"):
    dt = datetime.strptime(x, dt_format)
    return dt.strftime("%Y")

def get_series_by_year(data, year):
    dates = data['Date/Time']
    
    converted_dates = dates.map(get_year)
    
    mask = (converted_dates == str(year))
    
    return data.loc[mask]

def get_regressor(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        print('Loading existing regressor ----------------------------------')
        
        reg = LinearRegression()
        reg.intercept_ = weights[:,0]
        reg.coef_ = weights[:, 1:]

    return reg

def point_derivative(data):
    data['eColi_change'] = data['eColi'].diff()
    data.dropna(inplace=True)
    
    return data

class KalmanFilter():
    
    def __init__(self, F, x_0, Q_noise, R_noise, M, N, P_init):
        self.M = M
        self.N = N
        
        self.P = P_init
        
        self.Q_noise = Q_noise
        self.Q = Q_noise * np.eye(M)
        self.R = R_noise
        self.x = x_0
        self.F = F
        self.K = np.zeros((M, N))
        
    def predict(self):
        self.state_prediction()
        self.uncertainty_prediction()
        
    def state_prediction(self):
        self.x = self.F @ self.x # + np.random.normal(0, np.sqrt(self.Q_noise), size=(self.M, self.N))

    def uncertainty_prediction(self):
        self.P = (self.F @ self.P) @ self.F.T + self.Q
        
    def update(self, z, H):
        self.kalman_gain(H)
        self.state_update(z, H)
        self.uncertainty_update(H)
        
    def kalman_gain(self, H):
        self.K = self.P @ H.T @ np.linalg.pinv(H @ self.P @ H.T + self.R)
        
    def state_update(self, z, H):
        self.x = self.x + self.K @ (z - H @ self.x)
        
    def uncertainty_update(self, H):
        self.P = (np.eye(self.M) - self.K @ H) @ self.P

def kalman_forecast(df, input_features, Q_noise, R_noise, alpha):
    M = len(input_features)
    N = 1

    
    predictions = []
    true_labels = []
    uncertainty_log = []
    kalman_gain_log = []
    
    
    # file_path = "linear_transition_coeffs.pkl"
    # reg = get_regressor(file_path)
    
    np.random.seed(42)
    
    # initial state
    x = np.random.rand(M,N)
    # x = reg.coef_[0].T
    F = np.eye(M)    
    
    P = alpha * np.eye(M)
    
    
    kf = KalmanFilter(F, x, Q_noise, R_noise, M, N, P)

    df = df.set_index(np.arange(df.shape[0]))
    
    for i in range(df.shape[0]):
        
        H = np.array(df.iloc[i][input_features].copy().values, dtype=float).reshape(1,-1)
        
        kf.predict()
        
        predictions.append((H @ kf.x)[0])
        
        target = np.array(df.iloc[i]['eColi'].copy(), dtype=float)
        
        kf.update(target, H)
        
        true_labels.append(target)
        uncertainty_log.append(np.linalg.norm(kf.P))
        kalman_gain_log.append(np.linalg.norm(kf.K))
        
    return true_labels, predictions, uncertainty_log, kalman_gain_log
        
# I want to log and plot total uncertainty and Kalman gain over time
def per_year_forecast(data, Q_noise, R_noise, alpha):
    
    data = point_derivative(data)
    
    # yesterday's ecoli stuff and today's weather
    # these features are going to be the H vector
    input_features = ['eColi_prev'] + ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}
    
    # test_year = sorted(summer_data.keys())[-1]  # hold out last summer
    # test_set = summer_data.pop(test_year).copy()
    
    for year, df in summer_data.items():
        
        print(f'Forecasting {year}')
        
        df2 = df.copy(deep=True)
        
        df2['eColi_prev'] = df['eColi'].shift(1)
        df2['eColi_change_prev'] = df['eColi_change'].shift(1)
        df2.dropna(inplace=True)
        
        true_labels, predictions, uncertainty_log, kalman_gain_log = kalman_forecast(df2, input_features, Q_noise, R_noise, alpha)
        title = f'Kalman Forecasting for {year}'
    
        print(f'Year: {year}, MSE: {mean_squared_error(true_labels, predictions)}')
        plot_kalman(true_labels, predictions, uncertainty_log, kalman_gain_log, title)

# should probably reset state in-between summers 
def all_years_forecast(data, Q_noise, R_noise, alpha):
    data = point_derivative(data)
    
    df = data
    
    input_features = ['eColi_prev', 'eColi_change_prev'] + ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    
    df2 = df.copy(deep=True)
    
    df2['eColi_prev'] = df['eColi'].shift(1)
    df2['eColi_change_prev'] = df['eColi_change'].shift(1)
    df2.dropna(inplace=True)
    
    true_labels, predictions, uncertainty_log, kalman_gain_log = kalman_forecast(df2, input_features, Q_noise, R_noise, alpha)

    print(f'MSE: {mean_squared_error(true_labels, predictions)}')
    title = f'All years in one Kalman Forecasting'
    plot_kalman(true_labels, predictions, uncertainty_log, kalman_gain_log, title)
    
def classification_error(true_labels, predictions, threshold):
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    mask = np.argwhere(true_labels >= threshold)
    
    vals = predictions[mask]
    
    return 1 - len(np.argwhere(vals >= threshold)) / len(true_labels)
    
def tune_parameters(data, q_list, r_list, alpha_list):
    
    data = point_derivative(data)
    
    input_features = ['eColi_change_prev'] + ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 'Total Precip (mm)', 'Heat Deg Days (°C)', 'Cool Deg Days (°C)']
    
    summer_data = {year: get_series_by_year(data, year) for year in range(2007, 2025)}
    
    for year, df in summer_data.items():
        
        print(f'Forecasting {year}')
        
        best_score = float('inf')
        best_params = None
        best_labels = None
        best_predictions = None
        best_uncertainty_log = None
        best_kalman_gain_log = None
        

        for alpha, q, r in product(alpha_list, q_list, r_list):
        
            df2 = df.copy(deep=True)
            
            df2['eColi_prev'] = df['eColi'].shift(1)
            df2['eColi_change_prev'] = df['eColi_change'].shift(1)
            df2.dropna(inplace=True)
            
            try:
                true_labels, predictions, uncertainty_log, kalman_gain_log = kalman_forecast(df2, input_features, q, r, alpha)
                # mse = classification_error(true_labels, predictions, 30)
                mse = mean_squared_error(true_labels, predictions)
                if mse < best_score:
                    best_score = mse
                    best_params = (alpha, q, r)
                    best_labels = true_labels
                    best_predictions = predictions
                    best_uncertainty_log = uncertainty_log
                    best_kalman_gain_log = kalman_gain_log
            except Exception as e:
                print(f"Skipping combo alpha={alpha}, q={q}, r={r}: {e}")
                continue
            
        print(f"\n Best Params for {year}: alpha={best_params[0]}, Q={best_params[1]}, R={best_params[2]} with MSE={best_score:.4f}")
        plot_kalman(best_labels, best_predictions, best_uncertainty_log, best_kalman_gain_log, title=f"Summer {year} Best Kalman Filter alpha={best_params[0]}, Q={best_params[1]}, R={best_params[2]}")
    
        
# I should try tuning to maximize classification accuracy    
if __name__ == '__main__':
    data = pd.read_csv('water_safety/weather_data_scripts/cleaned_data/daily/cleaned_merged_toronto_city_hanlans.csv', index_col=0)
    # all_years_forecast(data, 0.001, 1, 0.1)
    # per_year_forecast(data, 0.001, 0.1, 20)
    
    tune_parameters(data, alpha_list=[0.1, 1, 0.5, 5, 10, 25, 35, 45, 75, 100], q_list=[1e-3, 1e-2, 1e-1, 0.5, 1], r_list=[1e-3, 1e-2, 1e-1, 0.5, 1])