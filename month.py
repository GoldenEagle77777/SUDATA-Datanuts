import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

df = pd.read_csv('data.csv')

#*************************************************** DATA EXTRACTION AND CLEANING ********************************************************

part1 = df.iloc[:, 11:20]
part2 = df.iloc[:, 21:22]
part3 = df.iloc[:, 22:27]

df = pd.concat([df['timestamp'], part1, part2, part3], axis=1)

cols_to_drop = ['delivery_time_deviation', 'risk_classification']
cols_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_existing)

df_clean = df.copy()
df_clean = df_clean.drop_duplicates(keep='first', ignore_index=True)
for col in df_clean.columns:
    if df_clean[col].dtype.kind in 'biufc': 
        med = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(med)
    else:
        df_clean[col] = df_clean[col].fillna('') 


numeric_cols = df_clean.select_dtypes(include=['number']).columns
scaler = StandardScaler()
numeric_cols = numeric_cols[0:10]
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])


df_clean['timestamp'] = pd.to_datetime(df['timestamp'])
df_clean.set_index('timestamp', inplace=True)
df_monthly = df_clean.resample('ME').mean()
print(df_monthly.shape)  
print(df_monthly.head)

#*************************************************** EXPLORATORY DATA ANALYSIS ********************************************************


plt.plot(df_monthly.index,df_monthly['port_congestion_level'],label='port_congestion_level',color='green')
plt.plot(df_monthly.index,df_monthly['shipping_costs'],label='shipping_costs',color='blue')
plt.xlabel('Time')
plt.ylabel('Factors')
plt.legend()
plt.show()
plt.close()

plt.plot(df_monthly.index,df_monthly['shipping_costs'],label='shipping_costs',color='blue')
plt.plot(df_monthly.index,df_monthly['route_risk_level'],label='route_risk_level',color='red')
plt.xlabel('Time')
plt.ylabel('Factors')
plt.legend()
plt.show()


#correlation matrix
corr = df_monthly.corr() 
plt.figure(figsize=(10, 8))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",vmin=-1, vmax=1, center=0,linewidths=0.5,square=True,cbar_kws={"shrink": 0.8, "label": "Correlation"})
plt.title("Correlation Heatmap of Variables")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

def poly_graph(t1, t2, degree=1):
    x = df_monthly[t1].values
    y = df_monthly[t2].values
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


print(poly_graph('iot_temperature','cargo_condition_status',1))
print(poly_graph('port_congestion_level','disruption_likelihood_score',1))


def time_decomposition(t):
    series = df_monthly[t]
    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=12) 
    decomposition.plot()
    plt.show()


for f in df.columns[1:]:time_decomposition(f)
print(df_monthly.columns)




