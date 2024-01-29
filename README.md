
### “**Efficient Time-series Similarity Measurement and Ranking Based on Anomaly Detection”**

## About

---

Wherever we are, the time series is there. and time series data is used in various fields. 

Here are some key areas:

**Finance**: Time series data such as stock prices, exchange rates, and interest rates play a crucial role in financial markets and institutions.

**Meteorology**: Weather data including temperature, precipitation, and wind speed is utilized for weather prediction and climate change analysis.

**Production and Manufacturing**: Time series data related to production levels, manufacturing line performance, and product quality is used for monitoring and optimizing production and manufacturing processes.

**Healthcare**: Time series data from patient biometric signals, medical devices, etc., is employed for disease diagnosis, monitoring treatment effectiveness, and predicting health conditions.

Time series similarity analysis involves assessing how similar two or more time series data are. This process may include comparing patterns, trends, cycles, etc., to identify similarities. 

<img src="https://github.com/choi-jiii/anomaly-timesim/assets/90977303/70f851e3-da22-488f-830b-2afc25595d39" width="500"> 

We propose an efficient method for measuring time series similarity that focuses on **anomalies** rather than the entire series.

In the end, we can find similar time series, even if they look different.

## How to run

---

### Configuration

-

### Parameters

Select the reference time series you want to analyze.

```python
ref = 'AAPL' # Reference time-series
df = stock[stock.Symbol == ref].iloc[:, :-1] # Except the 'Symbol' column (You should except the string columns)
```

Setting the parameters.

```python
"""
Parameter Setting
"""
pca_cnt = 2  # Number of principal components
contamination = 0.05 # Outlier rate(0~1, 1: whole time series)
```

Apply the robust scaling to data.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
```

Apply the PCA(Principal Component Analysis) to data.

```python
from sklearn.decomposition import PCA

pca = PCA()
df_scaled_pca = pca.fit_transform(df_scaled)
```

Select an anomaly detection model and apply it to data → **Isolation Forest** or **Autoencoder**

```python
# Isolation Forest
from sklearn.ensemble import IsolationForest

IF = IsolationForest(contamination=contamination).fit(df_scaled_pca)
anomaly_index, = np.where(IF.predict(df_scaled_pca) == -1)
anomaly_ts = df.index[anomaly_index] # # abnormal timestamp extraction
```

```python
# Autoencoder
from pyod.models.auto_encoder import AutoEncoder

AE = anomaly_detection_autoencoder(n_splits=4)
AE.fit(df_scaled_pca)
anomaly_ts = residual_sum.head(int(len(residual_sum) * contamination)).index
```

## Running the evaluation

---

We evaluate our method by measuring similarity twice.

First, we measure the similarity between anomalies and the entire time series using 4 evaluation metrics.

- **Pearson correlation coefficient**
- **Cosine similarity**
- **FastDTW**
- **SMAPE**

```python
def calculate_pearson_correlation_similarity(series1, series2):
    correlation_coefficient, p_value = pearsonr(series1, series2)
    return correlation_coefficient

def calculate_smape_similarity(series1, series2):
    return np.mean((np.abs(series1-series2))/(np.abs(series1)+np.abs(series2)))*100

def calculate_cosine_similarity(series1, series2):
    return 1 - cosine(series1, series2)

def calculate_fastdtw_similarity(series1, series2):
    distance, _ = fastdtw(series1, series2)
    return distance

def measure_similarity(df, anomaly_df):
    
    result = pd.DataFrame(index=df.columns)
    
    # # Zero Division prevent
    # df.replace(0, 1e-10, inplace=True)
    # anomaly_df.replace(0, 1e-10, inplace=True)
    
    for col in result.index:
        
        # Pearson
        pearson = calculate_pearson_correlation_similarity(df[ref], df[col])
        anomaly_pearson = calculate_pearson_correlation_similarity(anomaly_df[ref], anomaly_df[col])    
        
        # FastDTW
        distance = calculate_fastdtw_similarity(df[ref], df[col])
        anomaly_distance = calculate_fastdtw_similarity(anomaly_df[ref], anomaly_df[col])

        # SMAPE
        smape = calculate_smape_similarity(df[ref], df[col])
        anomaly_smape = calculate_smape_similarity(anomaly_df[ref], anomaly_df[col])

        # Cosine Similarity
        cosine_similarity = calculate_cosine_similarity(df[ref], df[col])
        anomaly_cosine_similarity = calculate_cosine_similarity(anomaly_df[ref], anomaly_df[col])
        
        result.at[col, 'pearson'] = pearson
        result.at[col, 'pearson_anomaly'] = anomaly_pearson
        
        result.at[col, 'Cosine'] = cosine_similarity
        result.at[col, 'Cosine_anomaly'] = anomaly_cosine_similarity

        result.at[col, 'FastDTW'] = distance
        result.at[col, 'FastDTW_anomaly'] = anomaly_distance 

        result.at[col, 'SMAPE'] = smape
        result.at[col, 'SMAPE_anomaly'] = anomaly_smape

    return result
```

Second, we use **Spearman's rank correlation coefficient** to measure the similarity based on the result of the first step.

```python
# Rank by Spearman's rank correlation coefficient 
result_rank_spearman = result_rank.corr(method='spearman')
```

You can get the result like this:

<img src="https://github.com/choi-jiii/anomaly-timesim/assets/90977303/c130b7f8-ab58-4720-a0c3-bdbba351b5ef" width="500"> 

You can also get the ranking. (The following figure shows the ranking case of FastDTW.)

<img src="https://github.com/choi-jiii/anomaly-timesim/assets/90977303/96115d2a-c54c-4cf6-8f90-394a6d7a7578" width="150"> 
