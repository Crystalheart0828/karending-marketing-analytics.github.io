## 1. Load data


```python
import pandas as pd
import numpy as np
```


```python
# Load the interactions CSV file
interactions_df = pd.read_csv('fake_data/interactions.csv')

# Load the conversion CSV file
conversion_df = pd.read_csv('fake_data/conversion.csv')
```

## 2. Data Wranggling


```python
# Merge the two dataframe
merged_df = pd.merge(interactions_df, conversion_df, on='ConversionID', how='left')
merged_df.head()
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
      <th>InteractionID</th>
      <th>TimeStamp_x</th>
      <th>UserID</th>
      <th>SessionID</th>
      <th>Channel</th>
      <th>InteractionType</th>
      <th>ConversionID</th>
      <th>ConversionValue</th>
      <th>TimeStamp_y</th>
      <th>ProductID</th>
      <th>ProductUnit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2021-10-17</td>
      <td>864</td>
      <td>c74e9605-c1ba-43b8-ba87-16cb7f2ba6ab</td>
      <td>Facebook</td>
      <td>Click</td>
      <td>1536.0</td>
      <td>4556.0</td>
      <td>2021-11-11</td>
      <td>118.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2021-02-12</td>
      <td>1618</td>
      <td>23112d4a-0c81-43ae-9cfb-964c92e25a0a</td>
      <td>Email</td>
      <td>Click</td>
      <td>1792.0</td>
      <td>7020.0</td>
      <td>2021-03-24</td>
      <td>108.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2021-06-21</td>
      <td>1409</td>
      <td>3121f6d1-07e8-4568-a63a-5c0b09d9258b</td>
      <td>Facebook</td>
      <td>Click</td>
      <td>144.0</td>
      <td>8987.0</td>
      <td>2021-03-13</td>
      <td>144.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2021-10-07</td>
      <td>2661</td>
      <td>6f36a6bf-acad-4854-bf07-e61804af559c</td>
      <td>Other Referrals</td>
      <td>Click</td>
      <td>1055.0</td>
      <td>6352.0</td>
      <td>2021-01-25</td>
      <td>150.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2021-08-19</td>
      <td>4483</td>
      <td>3680d88b-0641-485e-ac40-9768777772fe</td>
      <td>Instagram</td>
      <td>Click</td>
      <td>615.0</td>
      <td>9111.0</td>
      <td>2021-08-24</td>
      <td>137.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The Interaction that has a conversion should have the same timestamp

merged_df.loc[merged_df['ConversionID'].notna(), 'TimeStamp_y'] = merged_df['TimeStamp_x']
```


```python
# Identify users with conversions
users_with_conversion = merged_df[merged_df['ConversionID'].notna()]['UserID'].unique()

# Filter the DataFrame
converted_users_df = merged_df[merged_df['UserID'].isin(users_with_conversion)]

# Now, converted_users_df contains only data for users who have converted
```


```python
# Renaming columns in the DataFrame
converted_users_df = converted_users_df.rename(columns={
    'TimeStamp_x': 'TimeStamp_interaction',
    'TimeStamp_y': 'TimeStamp_conversion'
})
```


```python
# Assuming 'TimeStamp_interaction' and 'TimeStamp_conversion' columns are in your DataFrame

# Convert Timestamp columns to datetime
converted_users_df['TimeStamp_interaction'] = pd.to_datetime(converted_users_df['TimeStamp_interaction'])
converted_users_df['TimeStamp_conversion'] = pd.to_datetime(converted_users_df['TimeStamp_conversion'])

# Function to populate conversion timestamps for non-converted interactions
def populate_conversion_timestamps(group):
    group = group.sort_values('TimeStamp_interaction', ascending=False)
    
    # Forward fill to populate conversion timestamps for non-converted interactions
    group['TimeStamp_conversion'] = group['TimeStamp_conversion'].ffill()
    return group

# Apply the function to the DataFrame grouped by UserID
converted_users_df = converted_users_df.groupby('UserID').apply(populate_conversion_timestamps)

```


```python
# Calculate the time difference in days
converted_users_df['TimeDifference'] = (converted_users_df['TimeStamp_conversion'] - converted_users_df['TimeStamp_interaction']).dt.days

```


```python
# Filter out rows where the interaction timestamp is later than the conversion timestamp
filtered_df = converted_users_df[converted_users_df['TimeStamp_interaction'] <= converted_users_df['TimeStamp_conversion']]

```

## 3. Take a pause, check if the dataframe is as expected

1) Each user might have several interaction sessions. At least one of interactions leads to a conversion.
2) The conversion interaction is the latest interaction of that user.


```python
# Reset the index of the DataFrame
filtered_df = filtered_df.reset_index(drop=True)
filtered_df
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
      <th>InteractionID</th>
      <th>TimeStamp_interaction</th>
      <th>UserID</th>
      <th>SessionID</th>
      <th>Channel</th>
      <th>InteractionType</th>
      <th>ConversionID</th>
      <th>ConversionValue</th>
      <th>TimeStamp_conversion</th>
      <th>ProductID</th>
      <th>ProductUnit</th>
      <th>TimeDifference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>622</td>
      <td>2021-07-28</td>
      <td>3</td>
      <td>681f366d-30dc-40b1-aebe-271af9e42914</td>
      <td>YouTube</td>
      <td>Click</td>
      <td>851.0</td>
      <td>9208.0</td>
      <td>2021-07-28</td>
      <td>124.0</td>
      <td>20.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8718</td>
      <td>2021-03-21</td>
      <td>3</td>
      <td>30146a38-8956-4a22-8957-e3b7e2268898</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-07-28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2323</td>
      <td>2021-12-07</td>
      <td>7</td>
      <td>460d54f4-e6e1-4e84-8c63-7242c18336b3</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>2302.0</td>
      <td>6050.0</td>
      <td>2021-12-07</td>
      <td>120.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16293</td>
      <td>2021-07-25</td>
      <td>7</td>
      <td>cb7b5e7f-078b-4c81-81ac-b851461acf8a</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-12-07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13387</td>
      <td>2021-07-12</td>
      <td>7</td>
      <td>0df09c54-61f1-4912-8ca6-8723da5d0b8b</td>
      <td>Email</td>
      <td>View</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-12-07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>2023</td>
      <td>2021-05-14</td>
      <td>4985</td>
      <td>13f8fdb1-f1a9-4dcf-8661-6d6fe843ba6b</td>
      <td>Other Referrals</td>
      <td>Click</td>
      <td>2349.0</td>
      <td>4575.0</td>
      <td>2021-05-14</td>
      <td>125.0</td>
      <td>7.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>18947</td>
      <td>2021-04-29</td>
      <td>4985</td>
      <td>c08a488b-c782-4368-84e9-9b676146980e</td>
      <td>Email</td>
      <td>View</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-05-14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>2063</td>
      <td>2021-10-08</td>
      <td>4997</td>
      <td>fd8d7ab3-77ca-4873-94bd-cf23eafef31f</td>
      <td>Email</td>
      <td>Click</td>
      <td>1342.0</td>
      <td>9820.0</td>
      <td>2021-10-08</td>
      <td>111.0</td>
      <td>16.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>2475</td>
      <td>2021-03-29</td>
      <td>4997</td>
      <td>9035227f-a6b5-4bad-b8bc-eb0f6b1717ae</td>
      <td>Instagram</td>
      <td>View</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-10-08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>18976</td>
      <td>2021-02-21</td>
      <td>4997</td>
      <td>b422dd23-d840-4a53-9819-fe75d31eab4b</td>
      <td>Instagram</td>
      <td>Load</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-10-08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>229.0</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 12 columns</p>
</div>




```python
# Test 1: Each user should have at least one interaction leading to a conversion
users_with_conversion = filtered_df.groupby('UserID')['ConversionID'].apply(lambda x: x.notna().any())

# Use all() function to test if there's any False in the output
users_with_conversion.all()
```




    True




```python
# Test 2: The latest interaction for each user should be a conversion

def check_latest_interaction(group):
    latest_interaction = group.sort_values('TimeStamp_interaction', ascending=False).iloc[0]
    return pd.notna(latest_interaction['ConversionID'])

latest_interaction_is_conversion = filtered_df.groupby('UserID').apply(check_latest_interaction)

# Test if there's any Flase in the output
latest_interaction_is_conversion.all()
```




    True



## 4. Time-decay attribution model


```python
# Define the half-life period in days
half_life = 7

# Apply the exponential time-decay formula
filtered_df['DecayWeight'] = 0.5 ** (filtered_df['TimeDifference'] / half_life)

# Display the first few rows to verify the decay weight calculation
filtered_df.head()
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
      <th>InteractionID</th>
      <th>TimeStamp_interaction</th>
      <th>UserID</th>
      <th>SessionID</th>
      <th>Channel</th>
      <th>InteractionType</th>
      <th>ConversionID</th>
      <th>ConversionValue</th>
      <th>TimeStamp_conversion</th>
      <th>ProductID</th>
      <th>ProductUnit</th>
      <th>TimeDifference</th>
      <th>DecayWeight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>622</td>
      <td>2021-07-28</td>
      <td>3</td>
      <td>681f366d-30dc-40b1-aebe-271af9e42914</td>
      <td>YouTube</td>
      <td>Click</td>
      <td>851.0</td>
      <td>9208.0</td>
      <td>2021-07-28</td>
      <td>124.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8718</td>
      <td>2021-03-21</td>
      <td>3</td>
      <td>30146a38-8956-4a22-8957-e3b7e2268898</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-07-28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.0</td>
      <td>2.834309e-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2323</td>
      <td>2021-12-07</td>
      <td>7</td>
      <td>460d54f4-e6e1-4e84-8c63-7242c18336b3</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>2302.0</td>
      <td>6050.0</td>
      <td>2021-12-07</td>
      <td>120.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16293</td>
      <td>2021-07-25</td>
      <td>7</td>
      <td>cb7b5e7f-078b-4c81-81ac-b851461acf8a</td>
      <td>Google Search</td>
      <td>Click</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-12-07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>135.0</td>
      <td>1.564666e-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13387</td>
      <td>2021-07-12</td>
      <td>7</td>
      <td>0df09c54-61f1-4912-8ca6-8723da5d0b8b</td>
      <td>Email</td>
      <td>View</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-12-07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148.0</td>
      <td>4.318827e-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Normalize the decay weights within each user's journey
filtered_df['NormalizedDecayWeight'] = filtered_df.groupby('UserID')['DecayWeight'].transform(lambda x: x / x.sum())

```


```python
# Aggregate the normalized weights for each touchpoint across all users
channel_attribution = filtered_df.groupby('Channel')['NormalizedDecayWeight'].sum()

```


```python
# Sort and display the results to understand channel effectiveness
channel_attribution_sorted = pd.DataFrame(channel_attribution.sort_values(ascending=False))

channel_attribution_sorted['Percentage'] = channel_attribution_sorted['NormalizedDecayWeight']/channel_attribution_sorted['NormalizedDecayWeight'].sum()
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
      <th>NormalizedDecayWeight</th>
      <th>Percentage</th>
    </tr>
    <tr>
      <th>Channel</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Google Ads</th>
      <td>291.838018</td>
      <td>0.153680</td>
    </tr>
    <tr>
      <th>YouTube</th>
      <td>290.378026</td>
      <td>0.152911</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>288.136158</td>
      <td>0.151730</td>
    </tr>
    <tr>
      <th>Google Search</th>
      <td>267.972181</td>
      <td>0.141112</td>
    </tr>
    <tr>
      <th>Other Referrals</th>
      <td>261.429687</td>
      <td>0.137667</td>
    </tr>
    <tr>
      <th>Instagram</th>
      <td>256.281948</td>
      <td>0.134956</td>
    </tr>
    <tr>
      <th>Email</th>
      <td>242.963983</td>
      <td>0.127943</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate the contribution in revenue, in absolute numbers
totalrevenue = filtered_df['ConversionValue'].sum()
channel_attribution_sorted['RevenueContribution'] = channel_attribution_sorted['Percentage']*totalrevenue

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
      <th>NormalizedDecayWeight</th>
      <th>Percentage</th>
      <th>RevenueContribution</th>
    </tr>
    <tr>
      <th>Channel</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Google Ads</th>
      <td>291.838018</td>
      <td>0.153680</td>
      <td>1.848488e+06</td>
    </tr>
    <tr>
      <th>YouTube</th>
      <td>290.378026</td>
      <td>0.152911</td>
      <td>1.839240e+06</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>288.136158</td>
      <td>0.151730</td>
      <td>1.825040e+06</td>
    </tr>
    <tr>
      <th>Google Search</th>
      <td>267.972181</td>
      <td>0.141112</td>
      <td>1.697323e+06</td>
    </tr>
    <tr>
      <th>Other Referrals</th>
      <td>261.429687</td>
      <td>0.137667</td>
      <td>1.655883e+06</td>
    </tr>
    <tr>
      <th>Instagram</th>
      <td>256.281948</td>
      <td>0.134956</td>
      <td>1.623277e+06</td>
    </tr>
    <tr>
      <th>Email</th>
      <td>242.963983</td>
      <td>0.127943</td>
      <td>1.538922e+06</td>
    </tr>
  </tbody>
</table>
</div>



## 5. ROI Analysis


```python
campaign_df = pd.read_csv('fake_data/campaign_data.csv')
```


```python
channel_attribution_sorted['CampaignCost'] = campaign_df.groupby('Channel')['Cost'].sum()

channel_attribution_sorted['ROI'] = (channel_attribution_sorted['RevenueContribution']- channel_attribution_sorted['CampaignCost'])/ channel_attribution_sorted['CampaignCost']

(channel_attribution_sorted['.style
#                            .format(format_dict)
                           .highlight_max(color='#cd4f39')
                           .highlight_min(color='lightgreen'))
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[80], line 5
          1 channel_attribution_sorted['CampaignCost'] = campaign_df.groupby('Channel')['Cost'].sum()
          3 channel_attribution_sorted['ROI'] = (channel_attribution_sorted['RevenueContribution']- channel_attribution_sorted['CampaignCost'])/ channel_attribution_sorted['CampaignCost']
    ----> 5 (channel_attribution_sorted['ROI'].style
          6 #                            .format(format_dict)
          7                            .highlight_max(color='#cd4f39')
          8                            .highlight_min(color='lightgreen'))


    File ~/anaconda3/lib/python3.11/site-packages/pandas/core/generic.py:5902, in NDFrame.__getattr__(self, name)
       5895 if (
       5896     name not in self._internal_names_set
       5897     and name not in self._metadata
       5898     and name not in self._accessors
       5899     and self._info_axis._can_hold_identifiers_and_holds_name(name)
       5900 ):
       5901     return self[name]
    -> 5902 return object.__getattribute__(self, name)


    AttributeError: 'Series' object has no attribute 'style'

