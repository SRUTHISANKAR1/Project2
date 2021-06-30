# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:47:16 2021

@author: Shilpa Sujith
"""

import streamlit as st

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
sns.set()


pd.set_option('display.max_rows', 5000)

st.title("Retail Customer Predictor")

def get_data(suppress_st_warning = True):
 path = path = r'Retail-Ecommerce.xlsx'
 return pd.read_excel(path)

data = get_data()


# Data Cleaning

data['Sales'] = data ['Quantity'] * data ['UnitPrice'] 

data.drop_duplicates(keep=False, inplace=True)
remove_description = ['POSTAGE','DOTCOM POSTAGE','Bank Charges','AMAZON FEE','Next Day Carriage','PACKING CHARGE','Adjust bad debt']
data = data[~data['Description'].isin(remove_description)]

# Removing rows with unit price 0 or less and qauantity 0 or less. that implies to cancelled transactions
data_cleaned =  data[(data['UnitPrice']>0) & (data['Quantity']>0)]


#Dropping records with no customer id 
data_cleaned.dropna(inplace=True)

# Modelling Using RFM & Kmeans Clustering

import datetime as dt

PRESENT = dt.datetime(2011,12,10)
data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'])

# RFM Analysis - recency , frequency , monetary
RFM= data_cleaned.groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,
                                        'InvoiceNo': lambda InvoiceNo: InvoiceNo.nunique(),
                                        'Sales': lambda Sales: Sales.sum()})

# Change the name of columns
RFM.columns=['Recency','Frequency','Amount']

RFM.reset_index(level=[0], inplace=True)

# outlier treatment for Amount
plt.boxplot(RFM.Amount)
Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Amount >= (Q1 - 1.5*IQR)) & (RFM.Amount <= (Q3 + 1.5*IQR))]

# outlier treatment for Frequency
plt.boxplot(RFM.Frequency)
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]

# outlier treatment for Recency
plt.boxplot(RFM.Recency)
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]

# standardise all parameters
RFM_norm1 = RFM.drop(["CustomerID"], axis=1)
#RFM_norm1.Recency = RFM_norm1.Recency.dt.days

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)

RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Recency','Frequency','Amount']





# To perform KMeans clustering 
from sklearn.cluster import KMeans

list_k = list(range(2, 8))
sse = []
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(RFM_norm1)
    sse.append(km.inertia_)
    
# Kmeans with K=4
model_clus5 = KMeans(n_clusters = 4, max_iter=50)
model_clus5.fit(RFM_norm1)

pd.RangeIndex(len(RFM.index))

RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
RFM_km.columns = ['CustomerID','Recency', 'Frequency', 'Amount',  'ClusterID']




km_clusters_amount = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())


# analysis of clusters formed
RFM.index = pd.RangeIndex(len(RFM.index))
RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
RFM_km.columns = ['CustomerID', 'Recency','Frequency',  'Amount', 'ClusterID']


km_clusters_amount = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())

kmeans_df = pd.concat([pd.Series([0,1,2,3]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
kmeans_df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]

kmeans_df.sort_values(by=['Recency_mean','Frequency_mean'],inplace= True)

best_k_means_clust_id =  kmeans_df['ClusterID'].iloc[0]

#Top 10 customers - Sales

grouped_invoice = data_cleaned.groupby(['InvoiceNo','CustomerID','InvoiceDate','Country']).agg({'Quantity':'sum','Sales':'sum'})
grouped_invoice.reset_index(level=[0,1,2,3], inplace=True)

all_customer_sales  = grouped_invoice.groupby(['CustomerID']).agg({'Quantity':'sum','Sales':'sum'}).sort_values('Sales',ascending = False)
all_customer_sales.reset_index(level=[0], inplace=True)



# best customers
# RFM_km[RFM_km['ClusterID']==best_k_means_clust_id]


RFM_Best_Cluster_Kmeans = RFM_km[RFM_km['ClusterID']==best_k_means_clust_id]


RFM_Best_Cluster_Kmeans_products = RFM_Best_Cluster_Kmeans.merge(data_cleaned,how = 'inner', on = 'CustomerID')

# st.write(RFM_Best_Cluster_Kmeans =RFM_km[RFM_km['ClusterID']==best_k_means_clust_id]['Description'].value_counts().head(50))


## Combining RFM Best cluster from k means and cleaned data for sales info
RFM_Best_Cluster_Kmeans_product = RFM_Best_Cluster_Kmeans.merge(data_cleaned,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans_product['Invoice_Date'] = RFM_Best_Cluster_Kmeans_product['InvoiceDate'].dt.date

#create a dataframe with CustomerID and Invoice Date
tx_day_order = RFM_Best_Cluster_Kmeans_product[['CustomerID','InvoiceDate']]

#convert Invoice Datetime to day
tx_day_order['InvoiceDay'] = tx_day_order['InvoiceDate'].dt.date
tx_day_order = tx_day_order.sort_values(['CustomerID','InvoiceDate'])


#drop duplicates
tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')

#shifting last 3 purchase dates
tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(3)


tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
tx_day_order['DayDiff2'] = (tx_day_order['PrevInvoiceDate'] - tx_day_order['T2InvoiceDate']).dt.days
tx_day_order['DayDiff3'] = (tx_day_order['T2InvoiceDate'] - tx_day_order['T3InvoiceDate']).dt.days

tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']

tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')

RFM_Best_Cluster_Kmeans_product = RFM_Best_Cluster_Kmeans_product.merge(tx_day_diff,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans_product_Last_purchase = RFM_Best_Cluster_Kmeans_product.groupby('CustomerID').agg({
                                        'InvoiceDate': lambda  date: date.max()                    
                                                                    })

RFM_Best_Cluster_Kmeans_product_Last_purchase.columns = ['Last_purchase_date']

RFM_Best_Cluster_Kmeans = RFM_Best_Cluster_Kmeans.merge(RFM_Best_Cluster_Kmeans_product_Last_purchase,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans = RFM_Best_Cluster_Kmeans.merge(tx_day_diff,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans['Next_purchase_Date'] = RFM_Best_Cluster_Kmeans['Last_purchase_date'] +pd.to_timedelta(round(RFM_Best_Cluster_Kmeans['DayDiffMean']), unit='d')

RFM_Best_Cluster_Kmeans['Next_purchase_Date'] = RFM_Best_Cluster_Kmeans['Next_purchase_Date'].dt.date

RFM_Best_Cluster_Kmeans = RFM_Best_Cluster_Kmeans[(RFM_Best_Cluster_Kmeans.Next_purchase_Date >= date(2011,12,10)) 
                                          & (RFM_Best_Cluster_Kmeans.Next_purchase_Date < date(2012,1,31))]

RFM_Best_Cluster_Kmeans ['Expected_Amount'] = round(RFM_Best_Cluster_Kmeans['Amount']/RFM_Best_Cluster_Kmeans['Frequency'])

RFM_Best_Cluster_Kmeans = RFM_Best_Cluster_Kmeans.merge(all_customer_sales,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans ['Expected_Quantity'] = round(RFM_Best_Cluster_Kmeans['Quantity']/RFM_Best_Cluster_Kmeans['Frequency'])

RFM_Best_Cluster_Kmeans_final = RFM_Best_Cluster_Kmeans[['CustomerID','Recency','Frequency','Next_purchase_Date','Expected_Amount','Expected_Quantity']]

customer_products = RFM_Best_Cluster_Kmeans_product.groupby('CustomerID')['Description'].value_counts()

customer_products = RFM_Best_Cluster_Kmeans_product.groupby('CustomerID')['Description'].unique()
customer_products = pd.DataFrame(customer_products)
customer_products.reset_index(level=[0], inplace=True)
customer_products.set_index(['CustomerID'], inplace = True) 

RFM_Best_Cluster_Kmeans_final = RFM_Best_Cluster_Kmeans_final.merge(customer_products,how = 'inner', on = 'CustomerID')

RFM_Best_Cluster_Kmeans_final.rename(columns = {'Description':'Expected_Products'}, inplace = True)


# Sidebar = Frequency Range
min_frequency = int(RFM_Best_Cluster_Kmeans_final['Expected_Quantity'].min())
max_frequency = int(RFM_Best_Cluster_Kmeans_final['Expected_Quantity'].max())
selected_frquency_range = st.sidebar.slider('Select the Expected Quantity range', 0, max_frequency, (0, max_frequency), 1)


start_date = st.sidebar.date_input('Start date', date(2011,12,10))
end_date = st.sidebar.date_input('End date', date(2012,1,31))

if start_date >= end_date:
    st.error('Error: End date must fall after start date.')




# Sidebar = Amount Range
min_amount = int(RFM_Best_Cluster_Kmeans_final['Expected_Amount'].min())
max_amount = int(RFM_Best_Cluster_Kmeans_final['Expected_Amount'].max())
#sorted_unique_price = sorted(twsdata.Price.unique())
selected_amount_range = st.sidebar.slider('Select the Expected Purchase Amount range', 0, max_amount, (0, max_amount), 1)


RFM_Best_Cluster_Kmeans_final_filtered = RFM_Best_Cluster_Kmeans_final[
    (RFM_Best_Cluster_Kmeans_final['Expected_Quantity'] >= selected_frquency_range[0]) &  (RFM_Best_Cluster_Kmeans_final['Expected_Quantity'] <= selected_frquency_range[1]) &
    (RFM_Best_Cluster_Kmeans_final['Next_purchase_Date'] >= start_date) &  (RFM_Best_Cluster_Kmeans_final['Next_purchase_Date'] <= end_date) &
    (RFM_Best_Cluster_Kmeans_final['Expected_Amount'] >= selected_amount_range[0]) &  (RFM_Best_Cluster_Kmeans_final['Expected_Amount'] <= selected_amount_range[1])]

st.dataframe(RFM_Best_Cluster_Kmeans_final_filtered)





