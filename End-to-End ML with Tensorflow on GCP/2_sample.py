#!/usr/bin/env python
# coding: utf-8

# <h1> 2. Creating a sampled dataset </h1>
# 
# This notebook illustrates:
# <ol>
# <li> Sampling a BigQuery dataset to create datasets for ML
# <li> Preprocessing with Pandas
# </ol>

# In[1]:


# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'


# In[2]:


import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


# In[3]:


get_ipython().run_cell_magic('bash', '', 'if ! gsutil ls | grep -q gs://${BUCKET}/; then\n  gsutil mb -l ${REGION} gs://${BUCKET}\nfi')


# <h2> Create ML dataset by sampling using BigQuery </h2>
# <p>
# Let's sample the BigQuery data to create smaller datasets.
# </p>

# In[4]:


# Create SQL query using natality data after the year 2000
from google.cloud import bigquery
query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""


# There are only a limited number of years and months in the dataset. Let's see what the hashmonths are.

# In[5]:


# Call BigQuery but GROUP BY the hashmonth and see number of records for each group to enable us to get the correct train and evaluation percentages
df = bigquery.Client().query("SELECT hashmonth, COUNT(weight_pounds) AS num_babies FROM (" + query + ") GROUP BY hashmonth").to_dataframe()
print("There are {} unique hashmonths.".format(len(df)))
df.head()


# Here's a way to get a well distributed portion of the data in such a way that the test and train sets do not overlap:

# In[6]:


# Added the RAND() so that we can now subsample from each of the hashmonths to get approximately the record counts we want
trainQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) < 3 AND RAND() < 0.0005"
evalQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) = 3 AND RAND() < 0.0005"
traindf = bigquery.Client().query(trainQuery).to_dataframe()
evaldf = bigquery.Client().query(evalQuery).to_dataframe()
print("There are {} examples in the train dataset and {} in the eval dataset".format(len(traindf), len(evaldf)))


# <h2> Preprocess data using Pandas </h2>
# <p>
# Let's add extra rows to simulate the lack of ultrasound. In the process, we'll also change the plurality column to be a string.

# In[7]:


traindf.head()


# Also notice that there are some very important numeric fields that are missing in some rows (the count in Pandas doesn't count missing data)

# In[8]:


# Let's look at a small sample of the training data
traindf.describe()


# In[9]:


# It is always crucial to clean raw data before using in ML, so we have a preprocessing step
import pandas as pd
def preprocess(df):
  # clean up data we don't want to train on
  # in other words, users will have to tell us the mother's age
  # otherwise, our ML service won't work.
  # these were chosen because they are such good predictors
  # and because these are easy enough to collect
  df = df[df.weight_pounds > 0]
  df = df[df.mother_age > 0]
  df = df[df.gestation_weeks > 0]
  df = df[df.plurality > 0]
  
  # modify plurality field to be a string
  twins_etc = dict(zip([1,2,3,4,5],
                   ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']))
  df['plurality'].replace(twins_etc, inplace=True)
  
  # now create extra rows to simulate lack of ultrasound
  nous = df.copy(deep=True)
  nous.loc[nous['plurality'] != 'Single(1)', 'plurality'] = 'Multiple(2+)'
  nous['is_male'] = 'Unknown'
  
  return pd.concat([df, nous])


# In[10]:


traindf.head()# Let's see a small sample of the training data now after our preprocessing
traindf = preprocess(traindf)
evaldf = preprocess(evaldf)
traindf.head()


# In[11]:


traindf.tail()


# In[12]:


# Describe only does numeric columns, so you won't see plurality
traindf.describe()


# <h2> Write out </h2>
# <p>
# In the final versions, we want to read from files, not Pandas dataframes. So, write the Pandas dataframes out as CSV files. 
# Using CSV files gives us the advantage of shuffling during read. This is important for distributed training because some workers might be slower than others, and shuffling the data helps prevent the same data from being assigned to the slow workers.
# 

# In[13]:


traindf.to_csv('train.csv', index=False, header=False)
evaldf.to_csv('eval.csv', index=False, header=False)


# In[14]:


get_ipython().run_cell_magic('bash', '', 'wc -l *.csv\nhead *.csv\ntail *.csv')


# Copyright 2017-2018 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License

# In[ ]:




