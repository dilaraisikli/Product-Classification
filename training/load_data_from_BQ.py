#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.cloud import bigquery
client_bq = bigquery.Client.from_service_account_json("./credentials.json", project='charged-dialect-824')


# # Loading data from BQ

# In[3]:


def load_bq_data(_sql):
    _df = client_bq.query(_sql).to_dataframe()
    return _df


# In[4]:


sql = """
SELECT *
FROM RicardoInterview.product_detection_training_data
"""

df = load_bq_data(sql)
len(df)


# ## Save the Data

# In[5]:


df.to_csv('/home/ubuntu/product_classifier/di-interview-product-classifier/products/product_dataset.csv',index=False)

