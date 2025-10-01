import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('data.csv')

#******************************************* DATA EXTRACTION & CLEANING ********************************************************
part1 = df.iloc[:, 11:20]
part2 = df.iloc[:, 21:22]
part3 = df.iloc[:, 22:27]
df = pd.concat([df['timestamp'], part1, part2, part3], axis=1)


cols_to_drop = ['delivery_time_deviation', 'risk_classification']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])


df_clean = df.copy()
df_clean = df_clean.drop_duplicates(keep='first', ignore_index=True)
for col in df_clean.columns:
    if df_clean[col].dtype.kind in 'biufc': 
        med = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(med)
    else:
        df_clean[col] = df_clean[col].fillna('')

#Standardizing numeric features
numeric_cols = df_clean.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

#Convert 'timestamp' to datetime and set as index
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean.set_index('timestamp', inplace=True)

#Sample by week(starting on Monday)
df_weekly = df_clean.resample('W-MON').mean()

#******************************************* EXPLORATORY DATA ANALYSIS ********************************************************
plt.plot(df_weekly.index, df_weekly['port_congestion_level'], label='Port Congestion Level', color='green')
plt.plot(df_weekly.index, df_weekly['shipping_costs'], label='Shipping Costs', color='blue')
plt.xlabel('Time')
plt.ylabel('Factors')
plt.legend()
plt.title('Weekly Trends')
plt.show()

#Correlation matrix
corr = df_weekly.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8, "label": "Correlation"})
plt.title("Weekly Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Polynomial regression function
def poly_graph(t1, t2, degree=1):
    x = df_weekly[t1].values
    y = df_weekly[t2].values
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = p(x_line)
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x_line, y_line, color='red', label=f'y = ' + ' + '.join([f'{coeffs[i]:.3g} x^{degree-i}' for i in range(len(coeffs))]))
    plt.xlabel(t1)
    plt.ylabel(t2)
    plt.legend()
    plt.title(f'{t1} vs {t2}')
    plt.show()
    return coeffs

print(poly_graph('iot_temperature', 'cargo_condition_status', 1))
print(poly_graph('port_congestion_level', 'disruption_likelihood_score', 1))

#decomposing the time series
def time_decomposition(t):
    series = df_weekly[t]
    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=52)  # period=52 for weekly data
    decomposition.plot()
    plt.show()

for col in df_weekly.columns:
    time_decomposition(col)

#******************************************* DATA MODELLING ********************************************************

#look for stationarity
for col in df_weekly.columns:
    result = adfuller(df_weekly[col].dropna())
    print(f'{col}: ADF Statistic={result[0]}, p-value={result[1]}')

#handle non-stationary parts
df_weekly_diff = df_weekly.diff().dropna()

#find optimal lag length
model = VAR(df_weekly_diff)
lag_order = model.select_order().aic
print(f'Optimal Lag Length: {lag_order}')

#fit the Vector Auto Regression Model
model_fitted = model.fit(lag_order)

#predict values for the next 12 weeks
forecast_steps = 12
forecast = model_fitted.forecast(df_weekly_diff.values[-lag_order:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=pd.date_range(df_weekly.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W-MON'), columns=df_weekly.columns)
print(forecast_df)

#inverse transform forecasts if differenced
forecast_cumsum = forecast_df.cumsum()
forecast_original = df_weekly.iloc[-1] + forecast_cumsum
print(forecast_original)

#visualise_forecasts
for col in forecast_df.columns:
    plt.plot(forecast_df.index, forecast_df[col], label=col)

plt.title('Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Forecasted Value')
plt.legend()
plt.show()

#compare history and forecast
for col in df_weekly.columns:
    plt.figure()
    plt.plot(df_weekly.index, df_weekly[col], label="History")
    plt.plot(forecast_original.index, forecast_original[col], label="Forecast")
    plt.title(f"Forecast for {col}")
    plt.legend()
    plt.show()

