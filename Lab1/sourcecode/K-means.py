import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Import the data into a dataframe
df = pd.read_csv(Path('./College.csv'))
# To check the number of values in the private column
print(df["Private"].value_counts())
# check for the null values if they are present are not
nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

# There are no null counts in the data set so no need to take mean of the particular features

# Divide the dependent and independent data
y = df.iloc[:, 1:2]
x = df.iloc[:, 2:]
print(x.shape, y.shape)

# Elbow method is used for finding the optimal nnumber of clusters
wcss = []
for i in range(1, 6):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1,6),wcss)
plt.title('The Elbow Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Apply k mean clustering with no of clusters = 2
km = KMeans(n_clusters=2)
km.fit(x)
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("The silhouette_score is: ", score)

plt.scatter(x.iloc[:,10],x.iloc[:,5], c=y_cluster_kmeans[:,], s=30, cmap='viridis')


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Fit on training set only.
sc.fit(x)
# Apply transform to both the training set and the test set.
x_scaler = sc.transform(x)

# Apply PCA on the data with dimension reduction to 2 axis
pca = PCA(2)
x_pca = pca.fit_transform(x)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, df[['Private']]], axis=1)
print(finaldf)

# KMeans after PC
km = KMeans(n_clusters=3)
km.fit(x_pca)
y_cluster_kmeans= km.predict(x_pca)
from sklearn import metrics
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print("The silhouette_score after PCA is",score)

# Plotting the clusters
plt.scatter(x.iloc[:,10],x.iloc[:,5], c=y_cluster_kmeans[:,], s=50, cmap='viridis')
plt.show()