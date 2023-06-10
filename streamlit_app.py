!pip install matplotlib
from collections import namedtuple
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.cluster.hierarchy as sch
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


with st.echo(code_location='below'):
    dfgs = pd.read_excel("/content/Global Superstore (5000 data).xlsx")
    dfgshistory = dfgs.sort_values(by='Order Date').copy()
    
    dfgshistory['Order Date'].describe()

    dfgsuse = dfgs[['Row ID','Sales', 'Quantity', 'Discount', 'Profit']]

    def show_boxplot(df):
      plt.rcParams['figure.figsize'] = [14,6]
      sb.boxplot(data = df, orient="v")
      plt.title("Outliers Distribution", fontsize = 16)
      plt.ylabel("Range", fontweight = 'bold')
      plt.xlabel("Attributes", fontweight = 'bold')

    show_boxplot(dfgsuse)

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
    show_boxplot(dfgs_no_outl)

    data_scaler = StandardScaler()

    dfgs_scaled = data_scaler.fit_transform(dfgs_no_outl[['Sales', 'Quantity', 'Discount', 'Profit']])
    dfgs_scaled.shape

    complete_clustering = linkage(list(dfgs_scaled), method="complete", metric="euclidean")
    average_clustering = linkage(list(dfgs_scaled), method="average", metric="euclidean")
    single_clustering = linkage(list(dfgs_scaled), method="single", metric="euclidean")

    dendrogram(single_clustering)
    plt.show()

    dendrogram(average_clustering)
    plt.show()

    dendrogram(complete_clustering)
    plt.show()

    cluster_labels_c = sch.cut_tree(complete_clustering, n_clusters=2).reshape(-1, )
    dfgs_no_outl["Cluster"] = cluster_labels_c

    sb.boxplot(x='Cluster', y='Sales', data=dfgs_no_outl)

    sb.boxplot(x='Cluster', y='Quantity', data=dfgs_no_outl)

    sb.boxplot(x='Cluster', y='Profit', data=dfgs_no_outl)

    sb.boxplot(x='Cluster', y='Discount', data=dfgs_no_outl)

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
    dataset.info()

    dataset

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

    print("Data Penjualan Lima Teratas Produk yang Laku")
    datasetlaku_produk_info_sub_cat

    print("Data Penjualan Lima Teratas Sub-Katergori yang Laku")
    datasetlaku_subcattop5

    datasetlaku_subcattop5['Quantity'].plot(kind='bar')
    plt.title("Kuantitas Barang dari Lima Teratas Sub-Katergori yang Laku", fontsize=15, pad=15)

    dataset_laku.groupby('Sub-Category').agg({'Sales':'sum', 'Profit':'sum'}).sort_values(by=['Sales', 'Profit'], ascending=False).plot(kind='bar')
    plt.title("Perbandingan Sales dan Profit dari Sub-Katergori yang Laku", fontsize=15, pad=15)
    
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
