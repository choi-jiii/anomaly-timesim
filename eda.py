#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from pyod.models.auto_encoder import AutoEncoder
from sklearn.ensemble import IsolationForest

from tensorflow import keras
from sklearn.metrics import mean_squared_error

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from fastdtw import fastdtw


# In[2]:


class anomaly_detection_autoencoder:
    def __init__(self, n_splits=4):
        self.n_splits = n_splits
        self.reconstructed_data_lst = []
        self.residual_lst = []
        self.mse_scores = []
        self.fold_data = []

    def fit(self, input_array):
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(input_array), start=1):
            X_train, X_test = input_array[train_index], input_array[test_index]

            # 각 폴드의 데이터 저장 
            self.fold_data.append({
                'fold_idx': fold_idx,
                'train': (train_index[0], train_index[-1]),
                'test': (test_index[0], test_index[-1])
            })

            model = keras.Sequential([
                keras.layers.Input(shape=(input_array.shape[1],)),  # 입력 : 데이터 차원수
                keras.layers.Dense(30, activation='relu'),  # 인코더 레이어 (일반적으로 10~50)
                keras.layers.Dense(input_array.shape[1], activation='linear')  # 출력 : 입력 데이터 복원
            ])

            # 학습 프로세스 설정
            model.compile(optimizer='adam', loss='mean_squared_error')  # adam: 경사하강법 사용 알고리즘

            # 모델 학습
            model.fit(X_train, X_train, epochs=30, batch_size=32)

            # 모델 재구성
            reconstructed_data = model.predict(X_test)
            self.reconstructed_data_lst.append(reconstructed_data)

            # 원본 데이터와 재구성 데이터 간의 잔차 계산
            residual = X_test - reconstructed_data
            self.residual_lst.append(residual)

            # 평가 지표(MSE) 계산 및 결과 저장
            mse = mean_squared_error(X_test, reconstructed_data)
            self.mse_scores.append(mse)


# In[3]:


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


# ## Data definition

# In[4]:


# Define database
stock = pd.read_csv('./data/stock.csv', index_col=0)
stock.set_index('Date', inplace=True)


# In[5]:


# Select reference timeseries
ref = 'AAPL'
df = stock[stock.Symbol == ref].iloc[:, :-1] # Symbol 열 제거 (string)


# ## Data preprocessing

# ### Reference

# In[6]:


# Apply sacling to reference data
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)


# In[7]:


# Principal component analysis
pca = PCA()
df_scaled_pca = pca.fit_transform(df_scaled)


# In[8]:


# Check the principal component contribution Rate
pca_results = pd.DataFrame({'eigen_value': pca.explained_variance_,
                            'contribution_rate': pca.explained_variance_ratio_,
                            'cumulative_contribution_rate': pca.explained_variance_ratio_.cumsum(),})

pca_results.index = ['pca' + str(i) for i in range(1, len(pca_results) + 1)]

pca_contribute_lst = list(pca.explained_variance_ratio_) # Create contribution rate list
pca_results


# ### Subsequences

# In[9]:


stocks_scaled_pca_lst = []

for category, group in stock.groupby('Symbol'):
    group.drop('Symbol', axis=1, inplace=True) # string열 제외
    group_scaled = RobustScaler().fit_transform(group)
    group_scaled_pca = PCA().fit_transform(group_scaled)
    group_scaled_pca_df = pd.DataFrame(group_scaled_pca, index=group.index)
    group_scaled_pca_df['Symbol'] = category
    
    stocks_scaled_pca_lst.append(group_scaled_pca_df)

stocks_scaled_pca = pd.concat(stocks_scaled_pca_lst)


# ## Anomaly detection

# In[10]:


"""
Parameter Setting
"""
pca_cnt = 2  # Number of principal components
contamination = 0.05 # Anomaly contamination rate


# ### [Option: 1] Isolation Forest

# In[11]:


# Model Fitting
IF = IsolationForest(contamination=contamination).fit(df_scaled_pca)
anomaly_index, = np.where(IF.predict(df_scaled_pca) == -1)
anomaly_ts = df.index[anomaly_index] # # abnormal timestamp extraction


# ### [Option: 2] Autoencoder

# In[ ]:


AE = anomaly_detection_autoencoder(n_splits=4)
AE.fit(df_scaled_pca)

# Anomaly timestamp 추출 - 오차합 기반
residual_df = pd.DataFrame(np.vstack(AE.residual_lst), index=df.index) 
residual_sum = np.sum(residual_df, axis=1).sort_values(ascending=False) # 오차합
anomaly_ts = residual_sum.head(int(len(residual_sum) * contamination)).index


# ## Measuring similarity

# In[12]:


stocks_df = stocks_scaled_pca.pivot(columns='Symbol') # stocks with normal timestamp
stocks_anomaly_df = stocks_df[stocks_df.index.isin(anomaly_ts)] # stocks with abnormal timestamp

stocks_df_lst = list(map(lambda x: stocks_df[x], range(0, pca_cnt)))
stocks_df_anomaly_lst = list(map(lambda x: stocks_anomaly_df[x], range(0, pca_cnt)))


# In[13]:


# Measure similarity to normal and abnormal
result_lst = []
for i in range(0, pca_cnt): 
    result_lst.append(measure_similarity(stocks_df_lst[i], stocks_df_anomaly_lst[i]))


# In[14]:


# Apply contribution rate to the similarity of each principal component
result_weight_df = sum(map(lambda x: result_lst[x] * pca_contribute_lst[x], range(pca_cnt)))


# ## Spearman's rank correlaiton coefficient

# In[15]:


# Create the Rank Variable dataframe
result_rank = pd.concat([result_weight_df.iloc[:, :4].rank(ascending=False, method='first'),
                         result_weight_df.iloc[:,4:].rank(ascending=True)], axis=1)


# In[16]:


# Rank by Spearman's rank correlation coefficient 
result_rank_spearman = result_rank.corr(method='spearman')


# In[17]:


# Check performance
performance = result_rank_spearman.iloc[1::2, ::2]
performance


# In[18]:


recommend_rank_name = performance.max().idxmax() + '_anomaly'
recommend_rank = result_rank[recommend_rank_name]
recommend_rank

