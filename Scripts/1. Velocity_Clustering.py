'''
0. README
Notebook: Wasserstein_Distance_Velocity_Spectra
Inputs
•	pt_configuration_thesis.alltrucks_v1.combined_velocity_spectra_outlier_removed
•	pt_configuration_thesis.alltrucks_v1.weight_spectra_outliers
Outputs
•	pt_configuration_thesis.alltrucks_v1.velocity_cluster_results
Purpose
•	This notebook is meant to take as input the combined velocity spectra from Spectra_Outlier_Removal and weight_spectra_outliers from Weight_Spectra_Outlier_Detection.
•	It then proceeds to get rid of the outlier vins as described in readme section of Weight_Spectra_Outlier_Detction. This is followed by data preparation.
•	After this the notebook calculates Wasserstein distance to compare discrete spectra while respecting bin ordering and spacing. This has been done in a computationally effective manner.
•	The notebook then passes the velocity wasserstein distance to unsupervised k-medoids and segregates each vin into one of 3 clusters.
•	Sales group information is added back and the final output of this notebook is a dataframe which has 3 columns vin - corresponding sales group - corresponding velocity cluster assigned by k-medoids.
Other remarks
•	Note that the unsupervised clustering algorithm only gives you cluster information as a numeric integeer e.g. 0,1,2. Assigning & interpreting the clusters as High, Medium & Low should be done manually by looking at the visualization in section 10 of this notebook.
•	This is also potentially an area of improvement of this code.
'''


# 1.	HEADERS
import pyspark.sql.functions as f
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from functools import reduce

import datetime
import warnings

from talpy.timeseries import ts_transformer, ts_column_factory
from talpy.helper_functions import table_helper

# 2.	INPUT DATA
df_velocity_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`combined_velocity_spectra_outlier_removed`")

unique_vins_count_ori = df_velocity_spectra .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")
df_outliers = spark.table("pt_configuration_thesis.alltrucks_v1.`weight_spectra_outliers`")

unique_vins_count_ori = df_outliers.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")
outlier_vins = df_outliers.select("vin").distinct()
df_velocity_cleaned = df_velocity_spectra.join(outlier_vins, on="vin", how="left_anti")

unique_vins_count_ori = df_velocity_cleaned .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	MAKE DATA READY
from pyspark.sql.functions import lit

df_velocity_all = df_velocity_cleaned 

display(df_velocity_all)

# 4.	COMPUTE BIN CENTERS
# STEP 2 - COMPUTE BIN CENTER
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def get_bin_center(interval_str):
    cleaned = interval_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    left, right = cleaned.split(",")
    return (float(left.strip()) + float(right.strip())) / 2

get_bin_center_udf = udf(get_bin_center, DoubleType())

df_velocity_all = df_velocity_all.withColumn("bin_center", get_bin_center_udf("x1_interval"))

display(df_velocity_all)

# 5.	FIND GLOBAL MAXIMA & GENERATE ALL BIN CENTERS
# STEP 3 - COMPUTE GLOBAL MAXIMA
from pyspark.sql.functions import max as spark_max

global_max_velocity = df_velocity_all.agg(spark_max("bin_center")).collect()[0][0]

# global_max_velocity = 100  You can use this if you want to define a manual upper cap

print(f"Global max velocity across all trucks: {global_max_velocity}")

# STEP 4 - GENERATE ALL BIN CENTERS DEPENDING ON GLOBAL MAXIMA
import numpy as np

bin_interval = 1
all_bins = np.arange(0 + bin_interval/2, global_max_velocity + bin_interval, bin_interval)

# all_bins = np.arange(60 + bin_interval/2, global_max_velocity + bin_interval, bin_interval)

print(f"All bin centers: {all_bins}")

# 6.	ADD MISSING BINS FOR EACH TRUCK
# STEP 5 - Add missing bins for each truck. 

# Convert spark → pandas directly
df_pd = df_velocity_all.select("vin", "bin_center", "count").toPandas()

# Pivot to wide form
df_truck_distributions = df_pd.pivot_table(
    index="vin",
    columns="bin_center",
    values="count",
    aggfunc="sum",
    fill_value=0
)

# Ensure all bins exist
df_truck_distributions = df_truck_distributions.reindex(columns=all_bins, fill_value=0)

# Convert counts → probability
X = df_truck_distributions.to_numpy(dtype=float)
row_sums = X.sum(axis=1, keepdims=True)
X_probs = X / row_sums

bin_centers = df_truck_distributions.columns.values.astype(float)
vins = df_truck_distributions.index.values

print("X_probs shape:", X_probs.shape)
                              

# 7.	WASSERSTEIN DISTANCE CALCULATION
import numpy as np
from numba import njit, prange
import math
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
n_trucks = X_probs.shape[0]
bin_centers_array = bin_centers.astype(np.float64)
block_size = 1000  # block wise wasserstein distance to improve speed of the code

# -------------------------------
# NUMBA FUNCTION: 1D Wasserstein Distance Calculation 
# Numba JIT is used to reduce computational overhead in large-scale pairwise distance calculations
# I have tried my best to reduce several hours of computation to a few seconds. Maybe this can be further optimzized
# -------------------------------
@njit
def wasserstein_1d(u, v, bin_centers):
    """
    Exact 1D Wasserstein distance (EMD) between two discrete distributions u, v
    with same support given by bin_centers
    """
    cdf_u = np.cumsum(u)
    cdf_v = np.cumsum(v)
    distance = np.sum(np.abs(cdf_u - cdf_v) * np.diff(np.append(bin_centers, bin_centers[-1]+1)))
    return distance

# -------------------------------
# BLOCK-WISE DISTANCE MATRIX CALCULATION
# -------------------------------
distance_matrix = np.zeros((n_trucks, n_trucks), dtype=np.float64)

start_time = time.time()
print(f"Computing Wasserstein distance matrix for {n_trucks} trucks in blocks of {block_size}...")

# Precompute bin widths for efficiency
bin_widths = np.diff(np.append(bin_centers_array, bin_centers_array[-1]+1))

# Numba JIT for block computation
@njit(parallel=True)
def compute_block(X, start_i, end_i, distance_matrix, bin_widths):
    n = X.shape[0]
    for i in prange(start_i, end_i):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.sum(np.abs(np.cumsum(X[i]) - np.cumsum(X[j])) * bin_widths)
            distance_matrix[j, i] = distance_matrix[i, j]

# Loop over blocks
for start in range(0, n_trucks, block_size):
    end = min(start + block_size, n_trucks)
    compute_block(X_probs, start, end, distance_matrix, bin_widths)
    elapsed = time.time() - start_time
    print(f"Processed trucks {start} to {end} | Elapsed: {elapsed:.2f} s")

total_time = time.time() - start_time
print(f"Finished computing Wasserstein distance matrix in {total_time/60:.2f} minutes")

# -------------------------------
# distance_matrix is ready for k-medoids
# -------------------------------
print("Distance matrix shape:", distance_matrix.shape)

# Tip - please check if you have 1. zeroes along diagonal 2. symmetric numbers about diagonal 3. Min distance should be zero and never negative 4. Shape of matrix should be nXn where n is the number of trucks we considered

print("Distance matrix shape:", distance_matrix.shape)
print("Min distance:", distance_matrix.min())
print("Max distance:", distance_matrix.max())
print("Example distances (first 5 trucks):\n", distance_matrix[:5, :5])

# 8.	CLUSTERING
pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np

# Try different k to find the best silhouette
sil_scores = []
best_score = -1
best_labels = None
best_k = 2

# I have tried for 2 to 6 clusters as that is common practise. This can ofcourse be increased.
for k in range(2, 7):
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    sil_scores.append(score)
    print(f"k={k}, silhouette score={score:.4f}")

    # Save labels when k = 3
    if k == 3:
        labels_k3 = labels


# Note that we considered 3 clusters inspite of the fact that 2 clusters gave us a better score because as per discussion there is more operational sense in 3 clusters than 2. 

# Assign cluster labels for k = 3
df_truck_distributions['cluster_wasserstein'] = labels_k3

# 9.	CROSSTABS
# Add meta data i.e. sales group

# Get distinct VIN - Sales_Code_Group mapping
df_vin_group = df_velocity_all.select('vin', 'sales_group').distinct().toPandas()

# Make sure it is sorted the same as df_truck_distributions index (VIN)
df_vin_group = df_vin_group.set_index('vin').loc[df_truck_distributions.index]

# Add column
df_truck_distributions['sales_group'] = df_vin_group['sales_group'].values

crosstab = pd.crosstab(df_truck_distributions['cluster_wasserstein'],
                       df_truck_distributions['sales_group'])
print(crosstab)

display(crosstab)

# 10.	PLOTS FOR VISUALIZATIONS
# PCA for all number of clusters 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define cluster colors (can be freely chosen)
default_colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta']

# PCA projection once (same for all k)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_probs)

# Iterate over all k tested
for k_idx, k in enumerate(range(2, 7)):
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)
    
    # Assign colors (repeat colors if clusters > len(default_colors))
    colors = [default_colors[label % len(default_colors)] for label in labels]
    
    plt.figure(figsize=(10,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color=colors, s=50, alpha=0.8)
    
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title(f"PCA of Trucks Velocity Distributions (k={k} clusters)")
    
    # Custom legend
    legend_elements = [Patch(facecolor=default_colors[i], label=f'Cluster {i}') for i in range(k)]
    plt.legend(handles=legend_elements, title="Clusters")
    
    plt.grid(True, alpha=0.3)
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 1. Ensure cluster & sales_group exist
# -------------------------------
assert 'cluster_wasserstein' in df_truck_distributions.columns, "Cluster column missing"
assert 'sales_group' in df_truck_distributions.columns, "Sales group column missing"

# -------------------------------
# 2. Define consistent cluster colors
# -------------------------------
cluster_colors = {
    0: "#1f77b4",   # blue (medium velocity)
    1: "#ff7f0e",   # orange (high velocity)
    2: "#2ca02c"    # green (low velocity)
}

# -------------------------------
# 3. Define custom text labels
# -------------------------------
cluster_labels = {
    0: "Medium Velocity",
    1: "High Velocity",
    2: "Low Velocity"
}

# -------------------------------
# 4. List of sales groups
# -------------------------------
sales_groups = sorted(df_truck_distributions['sales_group'].unique())

# -------------------------------
# 5. Generate pie charts
# -------------------------------
num_groups = len(sales_groups)
cols = 3
rows = (num_groups + cols - 1) // cols

plt.figure(figsize=(18, 6 * rows))

for idx, sg in enumerate(sales_groups, 1):
    ax = plt.subplot(rows, cols, idx)

    subset = df_truck_distributions[df_truck_distributions['sales_group'] == sg]

    # Count trucks per cluster
    counts = subset['cluster_wasserstein'].value_counts().sort_index()

    # Ensure clusters in order 0,1,2
    counts = counts.reindex([0,1,2], fill_value=0)

    # Colors in fixed order
    colors = [cluster_colors[c] for c in counts.index]

    # Labels mapped to custom names
    labels = [cluster_labels[c] for c in counts.index]

    ax.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12}
    )
    ax.set_title(f"Sales Group {sg}: Velocity Cluster Distribution", fontsize=14)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Sales groups to plot (explicit order)
selected_sales_groups = [5, 8, 1]

plt.figure(figsize=(12, 4))

for idx, sg in enumerate(selected_sales_groups, 1):
    ax = plt.subplot(1, 3, idx)

    subset = df_truck_distributions[
        df_truck_distributions['sales_group'] == sg
    ]

    # Count trucks per cluster
    counts = subset['cluster_wasserstein'].value_counts().sort_index()

    # Ensure clusters [0, 1, 2] always exist
    counts = counts.reindex([0, 1, 2], fill_value=0)

    # Fixed colors
    colors = [cluster_colors[c] for c in counts.index]

    ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        autopct='%1.0f%%',
        textprops={
            'color': 'white',
            'fontsize': 18,
            'fontweight': 'bold'
        }
    )

    ax.set_title(
        f"SG {sg}",
        fontsize=24,
        fontweight='bold',
        color='red',
        pad=5
    )

plt.tight_layout()
plt.show()

# Visualize the 3 clusters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add cluster to probability matrix
X_probs_df = pd.DataFrame(X_probs, index=df_truck_distributions.index, columns=bin_centers)
X_probs_df['cluster'] = df_truck_distributions['cluster_wasserstein']

# Average distribution per cluster`
cluster_profiles = X_probs_df.groupby('cluster').mean()

print(cluster_profiles)


# Columns are bin centers
bin_centers = X_probs_df.columns[:-1].astype(float)

# Compute mean probability per bin for each cluster
cluster_profiles = X_probs_df.groupby('cluster').mean()

plt.figure(figsize=(10,6))

for cluster_label in cluster_profiles.index:
    plt.plot(bin_centers, cluster_profiles.loc[cluster_label, :], marker='o', markersize=4, label=f'Cluster {cluster_label}')

plt.xlabel("Velocity (kmph)")
plt.ylabel("Normalized Mileage")
plt.title("Average velocity distribution per Wasserstein cluster")
plt.legend()
plt.grid(True)
plt.show()

# 11.	MAKE DATA READY FOR CATALOG
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

# Reset index so VIN becomes a column
df_cluster_spark = df_truck_distributions.reset_index()[['vin', 'sales_group', 'cluster_wasserstein']]

# Optionally cast cluster to integer
df_cluster_spark['cluster_wasserstein'] = df_cluster_spark['cluster_wasserstein'].astype(int)

# Convert to Spark DataFrame
df_cluster_spark = spark.createDataFrame(df_cluster_spark)

# Show result
df_cluster_spark.show(10)
display(df_cluster_spark)

# 12.	PUSH ESSENTIAL DATA TO CATALOG
df_cluster_spark.write.mode("overwrite").saveAsTable("pt_configuration_thesis.alltrucks_v1.velocity_cluster_results")
