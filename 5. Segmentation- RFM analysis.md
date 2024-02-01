```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid")
```


```python
# Load the dataset
interaction_df = pd.read_csv("fake_data/segmentation_RFM/customer_interaction.csv") 
subscription_df = pd.read_csv("fake_data/segmentation_RFM/customer_subscription.csv") 

# Combine the two datasets
merge_df = pd.merge(interaction_df, subscription_df, on=["UserID","ContentType"], how="left")

# Drop rows that has ContentType conflicts
merge_df = merge_df.rename(columns={"ContentType":"ProductPurchased","TimeStamp":"LatestInteractionTime"})

# Turn columns into datetime objects
merge_df["LatestInteractionTime"] = pd.to_datetime(merge_df["LatestInteractionTime"])
merge_df["StartDate"] = pd.to_datetime(merge_df["StartDate"])
merge_df["EndDate"] = pd.to_datetime(merge_df["EndDate"])
```


```python
# Calculate inactive days, prep for following Recency analysis

import time

# Set a time
t = "2024-01-30"
current_time = pd.to_datetime(t)
merge_df['InactiveDays']= merge_df['LatestInteractionTime'].apply(lambda x: current_time-x).dt.days
```


```python
# Calculate accumlated interactions by userID, prep for following Frequency analysis

interaction_count = merge_df.groupby('UserID')['InteractionID'].count().to_dict()
merge_df['InteractionCounts']= merge_df['UserID'].map(interaction_count)
```


```python
# Calculate sum of revenue by UserID, prep for following revenue analysis

total_revenue = merge_df.groupby('UserID')['Revenue'].sum().to_dict()
merge_df['TotalRevenue']= merge_df['UserID'].map(total_revenue)
```

## RFM analysis


```python
# Calculate quantiles once outside the function
quantiles_recency = merge_df["InactiveDays"].quantile([0.2, 0.4, 0.6, 0.8])

# Write a function to classify the values in Series
def classify_recency(x, quantiles):
    if x > quantiles[0.8]:
        return 1
    elif quantiles[0.6] < x <= quantiles[0.8]:
        return 2
    elif quantiles[0.4] < x <= quantiles[0.6]:
        return 3
    elif quantiles[0.2] < x <= quantiles[0.4]:
        return 4
    else:
        return 5

# Apply the function to each element of the Series
merge_df["RecencyScore"] = merge_df["InactiveDays"].apply(lambda x: classify_recency(x, quantiles_recency))

```


```python
# Calculate quantiles once outside the function: frequency
quantiles_frequency = merge_df["InteractionCounts"].quantile([0.2, 0.4, 0.6, 0.8])

# Monetary
revenue_distribution = merge_df["TotalRevenue"].unique()
quantiles_monetary = pd.Series(revenue_distribution).quantile([0.2, 0.4, 0.6, 0.8])

def classify_frequency_revenue(y, quantiles):
    if y > quantiles[0.8]:
        return 5
    elif quantiles[0.6] < y <= quantiles[0.8]:
        return 4
    elif quantiles[0.4] < y <= quantiles[0.6]:
        return 3
    elif quantiles[0.2] < y <= quantiles[0.4]:
        return 2
    else:
        return 1

merge_df['FrequencyScore'] = merge_df["InteractionCounts"].apply(lambda y: classify_frequency_revenue(y, quantiles_frequency))
merge_df['MonetaryScore'] = merge_df["TotalRevenue"].apply(lambda y: classify_frequency_revenue(y, quantiles_monetary))

```

## Alternative: Using K-means Clustering for RFM analysis


```python
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```


```python
# Create a RFM dataframe for following analysis
rfm = merge_df[['UserID','InactiveDays','InteractionCounts','TotalRevenue']].sort_values(by='UserID')
rfm = rfm.groupby('UserID').min().reset_index()

# Check rfm dataframe dustribution
rfm_normalized = rfm[['InactiveDays','InteractionCounts','TotalRevenue']]
rfm_normalized.boxplot() 
```




    <Axes: >




    
![png](output_10_1.png)
    



```python
scaler = StandardScaler()
rfm_normalized = scaler.fit_transform(rfm_normalized)
rfm_normalized = pd.DataFrame(rfm_normalized)

# rfm_normalized.boxplot()
rfm_normalized.rename(columns={0:'Recency',1:'Frequency',2:'Monetary'})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.013384</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.815623</td>
      <td>0.518594</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.139070</td>
      <td>0.518594</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.495848</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.107061</td>
      <td>1.300497</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4283</th>
      <td>0.901779</td>
      <td>0.518594</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>4284</th>
      <td>-1.221554</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>4285</th>
      <td>-0.794806</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>4286</th>
      <td>-1.086244</td>
      <td>0.518594</td>
      <td>-0.27178</td>
    </tr>
    <tr>
      <th>4287</th>
      <td>-0.399283</td>
      <td>0.518594</td>
      <td>-0.27178</td>
    </tr>
  </tbody>
</table>
<p>4288 rows × 3 columns</p>
</div>




```python
# List to store silhouette scores
silhouette_scores = []
ssd = []
range_n_clusters = [2,3,4,5,6,7,8]

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = num_clusters, max_iter=50)
    kmeans.fit(rfm_normalized)
    
    cluster_labels = kmeans.labels_
    
    #silhouette score
    silhouette_avg = silhouette_score(rfm_normalized, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

```

    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=2, the silhouette score is 0.34968598877022017


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=3, the silhouette score is 0.403149149011258


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=4, the silhouette score is 0.4121058424640369


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=5, the silhouette score is 0.4216890863018657


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=6, the silhouette score is 0.4149176067009878


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=7, the silhouette score is 0.3831774697248856


    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(


    For n_clusters=8, the silhouette score is 0.40129503823652773



```python
import matplotlib.pyplot as plt

plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Counts')
plt.show()
```


    
![png](output_13_0.png)
    



```python
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_normalized)
```

    /Users/KarenDing/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(max_iter=50, n_clusters=4)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(max_iter=50, n_clusters=4)</pre></div></div></div></div></div>




```python
rfm_normalized.loc[:,'UserID'] = rfm['UserID']
rfm_normalized = rfm_normalized.rename(columns={0:'Recency',1:'Frequency',2:'Monetary'})
```


```python
rfm_normalized['Cluster'] = kmeans.labels_
rfm_normalized
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>UserID</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.013384</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.815623</td>
      <td>0.518594</td>
      <td>-0.27178</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.139070</td>
      <td>0.518594</td>
      <td>-0.27178</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.495848</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.107061</td>
      <td>1.300497</td>
      <td>-0.27178</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4283</th>
      <td>0.901779</td>
      <td>0.518594</td>
      <td>-0.27178</td>
      <td>4996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4284</th>
      <td>-1.221554</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
      <td>4997</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4285</th>
      <td>-0.794806</td>
      <td>-0.263309</td>
      <td>-0.27178</td>
      <td>4998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4286</th>
      <td>-1.086244</td>
      <td>0.518594</td>
      <td>-0.27178</td>
      <td>4999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4287</th>
      <td>-0.399283</td>
      <td>0.518594</td>
      <td>-0.27178</td>
      <td>5000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4288 rows × 5 columns</p>
</div>




```python
sns.boxplot(x='Cluster',y='Recency',data=rfm_normalized )
```




    <Axes: xlabel='Cluster', ylabel='Recency'>




    
![png](output_17_1.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set the color palette
palette = sns.color_palette("hsv", len(rfm_normalized['Cluster'].unique()))

# Plotting Recency vs. Frequency
plt.figure(figsize=(10, 6))
for cluster in rfm_normalized['Cluster'].unique():
    # Filter the data for one cluster
    cluster_data = rfm_normalized[rfm_normalized['Cluster'] == cluster]
    
    # Plot the data for this cluster
    plt.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                color=palette[cluster], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Scatter Plot of Recency vs Frequency by Cluster')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.legend()
plt.show()

```


    
![png](output_18_0.png)
    


<li>Customers with high recency and high frequency are often the most valuable; they engage often and have done so recently.</li>
<li>Customers with high frequency but low recency may be lapsing and could be targets for re-engagement campaigns.</li>
<li>Customers with low frequency and high recency are often new customers who might be encouraged to engage more often.</li>


```python
# Plotting Recency vs. Monetary

plt.figure(figsize=(10, 6))
for cluster in rfm_normalized['Cluster'].unique():
    # Filter the data for one cluster
    cluster_data = rfm_normalized[rfm_normalized['Cluster'] == cluster]
    
    # Plot the data for this cluster
    plt.scatter(cluster_data['Recency'], cluster_data['Monetary'], 
                color=palette[cluster], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Scatter Plot of Recency vs Monetary by Cluster')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend()
plt.show()

```


    
![png](output_20_0.png)
    


<li>Customers with high recency and high monetary value are likely to be big spenders and recent buyers, making them prime targets for upselling or cross-selling.</li>
<li>Customers with low recency but high monetary value may have been valuable in the past but are at risk of churning.</li>


```python
# Plotting Frequency vs. Monetary

plt.figure(figsize=(10, 6))
for cluster in rfm_normalized['Cluster'].unique():
    # Filter the data for one cluster
    cluster_data = rfm_normalized[rfm_normalized['Cluster'] == cluster]
    
    # Plot the data for this cluster
    plt.scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                color=palette[cluster], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Scatter Plot of Frequency vs Monetary by Cluster')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend()
plt.show()

```


    
![png](output_22_0.png)
    


<li>Customers with both high frequency and high monetary value are your most loyal and high-spending customers.</li>
<li>Customers with high frequency but low monetary value might be frequent buyers of lower-value items; there might be an opportunity to upsell higher-value products to them.</li>

### Following analysis that could be done

<li><b>Segment Profiling:</b>

Develop detailed profiles for each cluster, including their preferences, behaviors, and demographics.
Identify the most profitable segments for targeted marketing strategies.</li>
</br>
<li><b>Customer Lifecycle Analysis:</b>

Look at the customer lifecycle stages of each segment. Tailor marketing and service efforts to move customers toward higher-value segments.</li>
</br>
<li><b>Predictive Modeling:</b>

Use the clusters as inputs for predictive models to forecast future customer behavior, lifetime value, or the likelihood of churn.</li>
</br>
<li><b>A/B Testing:</b>

Test different engagement strategies for each segment to see what works best in terms of increasing frequency, recency, and monetary value.</li>
</br>
<li><b>Cross-Channel Analysis:</b>

Examine the role of different marketing channels within each segment to optimize channel strategy and budget allocation.</li>
</br>
<li><b>Customer Feedback Collection:</b>

Gather qualitative feedback from each segment to gain insights into their needs and improve product offerings.</li>
</br>

## Reference
1. <a href=https://medium.com/web-mining-is688-spring-2021/exploring-customers-segmentation-with-rfm-analysis-and-k-means-clustering-118f9ffcd9f0>Exploring Customers Segmentation With RFM Analysis and K-Means Clustering</a>
