import streamlit as st 
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Header
st.title("PREDICTING DATA : LAKU DAN TIDAK LAKU")
st.header("Mata kuliah : Perancangan Aplikasi Sains Data")
st.write(" 1. Alya Selynindya (1305210079) \n 2. Shamaya Mayra Argyanti(1305213112) \n 3. Jati Tepatasa Bagastakwa(1305213059)" )
st.text(" - Aplikasi sains data untuk memprediksi laku dan tidak laku") 
st.text("   sebuah barang dari dataset global superstore")

#menampilkan boxplot
dfgs = pd.read_excel(r"C:\Users\alsel\Desktop\Global Superstore (5000 data).xlsx")

dfgsuse = dfgs[['Row ID','Sales', 'Quantity', 'Discount', 'Profit']]
if(st.button("Dataset Global Superstore")) :
   dfgsuse

#------------------------------------------------------------------------------------------------------

with st.container() :
  st.write("---")

st.write( "Boxplot Outliers Distribution" )
def show_boxplot(df):
    fig, ax = plt.subplots(figsize=[14,6])
    sb.boxplot(data=df, orient="v", ax=ax)
    ax.set_title("Outliers Distribution", fontsize=16)
    ax.set_ylabel("Range", fontweight='bold')
    ax.set_xlabel("Attributes", fontweight='bold')
    st.pyplot(fig)

if(st.button("Visualisasi Boxplot")) :
   show_boxplot(dfgsuse)

#------------------------------------------------------------------------------------------------------

st.write( "Boxplot Outliers Distribution" )
def remove_outliers(data):
  df = data.copy() 
  for col in list(df.columns):
    if col != "Row ID":
      Q1 = df[str(col)].quantile(0.05)
      Q3 = df[str(col)].quantile(0.95)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5*IQR
      upper_bound = Q3 + 1.5*IQR
      df = df[(df[str(col)] >= lower_bound) & (df[str(col)] <= upper_bound)]
  return df

dfgs_no_outl = remove_outliers(dfgsuse)

if(st.button("Visualisasi Boxplot dengan Percentil")) :
   show_boxplot(dfgs_no_outl)
   dfgs_no_outl

#------------------------------------------------------------------------------------------------------

with st.container() :
  st.write("---")

from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
dfgs_scaled = data_scaler.fit_transform(dfgs_no_outl[['Sales', 'Quantity', 'Discount', 'Profit']])

from scipy.cluster.hierarchy import linkage, dendrogram

complete_clustering = linkage(list(dfgs_scaled), method="complete", metric="euclidean")
average_clustering = linkage(list(dfgs_scaled), method="average", metric="euclidean")
single_clustering = linkage(list(dfgs_scaled), method="single", metric="euclidean")

st.write("Single Clustering")
fig, ax = plt.subplots(figsize=[10, 6])
dendrogram(single_clustering, ax=ax)
ax.set_title("Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

#------------------------------------------------------------------------------------------------------

st.write("Average Clustering")
fig, ax = plt.subplots(figsize=[10, 6])
dendrogram(average_clustering, ax=ax)
ax.set_title("Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

#------------------------------------------------------------------------------------------------------

st.write("Complete Clustering")
fig, ax = plt.subplots(figsize=[10, 6])
dendrogram(complete_clustering, ax=ax)
ax.set_title("Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

#------------------------------------------------------------------------------------------------------

import scipy.cluster.hierarchy as sch
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

st.write("Boxplot Cluster Sales")
cluster_labels_c = sch.cut_tree(complete_clustering, n_clusters=2).reshape(-1, )
dfgs_no_outl["Cluster"] = cluster_labels_c
sb.boxplot(x='Cluster', y='Sales', data=dfgs_no_outl)
st.pyplot()

st.write("Boxplot Cluster Quantity")
sb.boxplot(x='Cluster', y='Quantity', data=dfgs_no_outl)
st.pyplot()

st.write("Boxplot Cluster Profit")
sb.boxplot(x='Cluster', y='Profit', data=dfgs_no_outl)
st.pyplot()

st.write("Boxplot Cluster Discount")
sb.boxplot(x='Cluster', y='Discount', data=dfgs_no_outl)
st.pyplot()

#------------------------------------------------------------------------------------------------------

dfgs_no_outl.reset_index(inplace=True)
dfgs_noidxclm = dfgs_no_outl.drop(columns=['index'])

dfgs_end = pd.merge(dfgs, dfgs_noidxclm, on="Row ID")
dataset = dfgs_end.drop(['Sales_y','Quantity_y', 'Profit_y', 'Discount_y'], axis=1)
dataset.reset_index()
dataset = dataset[['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 'Country',
       'Market', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales_x', 'Quantity_x', 'Discount_x', 'Profit_x',
       'Shipping Cost', 'Order Priority', 'Cluster']]
dataset.columns = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 'Country',
       'Market', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit',
       'Shipping Cost', 'Order Priority', 'Label']
#dataset.info()
#dataset

feature_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
X = dataset[feature_cols] # Features
y = dataset["Label"] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

st.write("Akurasi :")
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

y_test.value_counts()

dataset_laku = dataset[dataset["Label"]==1]
dataset_laku.reset_index(inplace=True)
dataset_laku = dataset_laku[['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 'Country',
       'Market', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit',
       'Shipping Cost', 'Order Priority', 'Label']]
dataset_tdklaku = dataset[dataset["Label"]==0]
dataset_tdklaku.reset_index(inplace=True)
dataset_tdklaku = dataset_tdklaku[['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 'Country',
       'Market', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit',
       'Shipping Cost', 'Order Priority', 'Label']]
for produk in pd.Series(dataset_tdklaku['Product Name'].unique()):
  if (produk in pd.Series(dataset_laku['Product Name'].unique())) == True:
    dataset_laku.drop(dataset_laku['Product Name']==produk)

dataset_laku = dataset_laku[dataset_laku["Label"]==1]

datasetlaku_copy = dataset_laku.copy()
datasetlaku_subcattop5 = datasetlaku_copy.groupby('Sub-Category').agg({'Sales':'sum', 'Quantity':'sum', 'Profit':'sum'}
                                                                  ).sort_values(by=['Quantity', 'Sales', 'Profit'], ascending=False).head(5)
datasetlaku_produk_info_sub_cat = datasetlaku_copy.groupby(['Product Name', 'Sub-Category']).agg({'Sales':'sum', 'Quantity':'sum', 'Profit':'sum'}
                                                               ).sort_values(by=['Quantity', 'Sales', 'Profit'], ascending=False).head(5)

if(st.button("Dataset Laku : ")) :
   dataset_laku

st.write("Data Penjualan Lima Teratas Produk yang Laku")
datasetlaku_produk_info_sub_cat

st.write("Data Penjualan Lima Teratas Sub-Katergori yang Laku")
datasetlaku_subcattop5

datasetlaku_subcattop5['Quantity'].plot(kind='bar')
st.write("Kuantitas Barang dari Lima Teratas Sub-Katergori yang Laku")
plt.title("Kuantitas Barang dari Lima Teratas Sub-Katergori yang Laku", fontsize=15, pad=15)

dataset_laku.groupby('Sub-Category').agg({'Sales':'sum', 'Profit':'sum'}).sort_values(by=['Sales', 'Profit'], ascending=False).plot(kind='bar')
plt.title("Perbandingan Sales dan Profit dari Sub-Katergori yang Laku", fontsize=15, pad=15)
