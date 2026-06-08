'''
0. README
Notebook: Wasserstein_Distance_Gradient_Spectra
Inputs
•	pt_configuration_thesis.alltrucks_v1.combined_gradient_spectra_outlier_removed
•	pt_configuration_thesis.alltrucks_v1.weight_spectra_outliers
Outputs
•	pt_configuration_thesis.alltrucks_v1.gradient_cluster_results
Purpose
•	This notebook is meant to take as input the combined gradient spectra from Spectra_Outlier_Removal and weight_spectra_outliers from Weight_Spectra_Outlier_Detection.
•	It then proceeds to get rid of the outlier vins as described in readme section of Weight_Spectra_Outlier_Detction. This is followed by data preparation.
•	After this the notebook calculates Wasserstein distance to compare discrete spectra while respecting bin ordering and spacing. This has been done in a computationally effective manner.
•	The notebook then passes the gradient wasserstein distance to unsupervised k-medoids and segregates each vin into one of 2 clusters.
•	Sales group information is added back and the final output of this notebook is a dataframe which has 3 columns vin - corresponding sales group - corresponding gradient cluster assigned by k-medoids.
Other remarks
•	Note that the unsupervised clustering algorithm only gives you cluster information as a numeric integeer e.g. 0 or 1. Assigning & interpreting the clusters as Hilly & Flat should be done manually by looking at the visualization in section 10 of this notebook.
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
df_gradient_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`combined_gradient_spectra_outlier_removed`")

unique_vins_count_ori = df_gradient_spectra .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

df_outliers = spark.table("pt_configuration_thesis.alltrucks_v1.`weight_spectra_outliers`")

unique_vins_count_ori = df_outliers.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

outlier_vins = df_outliers.select("vin").distinct()
df_gradient_cleaned = df_gradient_spectra.join(outlier_vins, on="vin", how="left_anti")

unique_vins_count_ori = df_gradient_cleaned .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	MAKE DATA READY
# STEP 1 - JOIN VECTORS 
from pyspark.sql.functions import lit

df_gradient_all = df_gradient_cleaned

display(df_gradient_all)

# 4.	COMPUTE BIN CENTERS
# STEP 2 - COMPUTE BIN CENTER
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def get_bin_center(interval_str):
    cleaned = interval_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    left, right = cleaned.split(",")
    return (float(left.strip()) + float(right.strip())) / 2

get_bin_center_udf = udf(get_bin_center, DoubleType())

df_gradient_all = df_gradient_all.withColumn("bin_center", get_bin_center_udf("x1_interval"))

display(df_gradient_all)

# 5.	FIND GLOBAL MAXIMA
# STEP 3 - COMPUTE GLOBAL MAXIMA
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min

global_max_gradient = df_gradient_all.agg(spark_max("bin_center")).collect()[0][0]
print(f"Global max gradient across all trucks: {global_max_gradient}")

global_min_gradient = df_gradient_all.agg(spark_min("bin_center")).collect()[0][0]
print(f"Global min gradient across all trucks: {global_min_gradient}")

# STEP 4 - GENERATE ALL BIN CENTERS DEPENDING ON GLOBAL MAXIMA
import numpy as np

bin_interval = 1
# all_bins = np.arange(0 + bin_interval/2, global_max_velocity + bin_interval, bin_interval)

# all_bins = np.arange(12000 + bin_interval/2, 49000 + bin_interval, bin_interval)

# all_bins = np.arange(global_min_gradient + bin_interval, global_max_gradient + bin_interval, bin_interval)

# all_bins = np.arange(-7.5 + bin_interval, 6.5 + bin_interval, bin_interval)

all_bins = np.arange(-0.5 + bin_interval, 9.5 + bin_interval, bin_interval)


print(f"All bin centers: {all_bins}")

# 6.	ADD MISSING BINS FOR EACH TRUCK
# -------------------------------
# STEP 5 - Add missing bins for each truck (gradient spectra)
# -------------------------------

# Convert Spark DataFrame → Pandas
df_pd = df_gradient_all.select("vin", "bin_center", "count").toPandas()

# Pivot to wide form: rows=trucks, columns=bin centers
df_truck_distributions = df_pd.pivot_table(
    index="vin",
    columns="bin_center",
    values="count",
    aggfunc="sum",
    fill_value=0
)

# Ensure all bins exist (fill missing bins with 0)
df_truck_distributions = df_truck_distributions.reindex(columns=all_bins, fill_value=0)

# Convert counts → probability distribution
X = df_truck_distributions.to_numpy(dtype=float)
row_sums = X.sum(axis=1, keepdims=True)
X_probs = X / row_sums

# Extract bin centers and VINs
bin_centers = df_truck_distributions.columns.values.astype(float)
vins = df_truck_distributions.index.values

# Print shape of probability matrix
print("X_probs shape:", X_probs.shape)

import numpy as np

# -------------------------------
# Identify all-zero rows using raw counts X
# -------------------------------
row_sums_before = X.sum(axis=1)  # sum across bins for each truck
zero_rows = np.where(row_sums_before == 0)[0]  # indices of all-zero rows

# Display rows that will be removed
print("Rows removed (all-zero distributions):", zero_rows)

# Total number of rows removed
print("Total rows removed:", len(zero_rows))

# Remove all-zero rows from X, X_probs, vins, and df_truck_distributions
X = np.delete(X, zero_rows, axis=0)
vins = np.delete(vins, zero_rows, axis=0)
df_truck_distributions = df_truck_distributions.drop(df_truck_distributions.index[zero_rows])

# Recompute probability distributions for remaining trucks
row_sums = X.sum(axis=1, keepdims=True)
X_probs = X / row_sums

# Display new shape
print("New X_probs shape:", X_probs.shape)

# 7.	WASSERSTEIN DISTANCE CALCULATION
import numpy as np
from numba import njit, prange
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
    with the same support given by bin_centers
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
# distance_matrix is ready for K-Medoids clustering
# -------------------------------
print("Distance matrix shape:", distance_matrix.shape)
print("Min distance:", distance_matrix.min())
print("Max distance:", distance_matrix.max())
print("Example distances (first 5 trucks):\n", distance_matrix[:5, :5])

# 8.	CLUSTERING
pip install scikit-learn-extra
'''
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np

# -------------------------------
# Try different k to find the best silhouette
# -------------------------------
sil_scores = []
best_score = -1
best_labels = None
best_k = None

# Example range: 2 to 6 clusters
for k in range(2, 7):
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)
    
    # Silhouette score using precomputed distance
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    sil_scores.append(score)
    
    print(f"k={k}, silhouette score={score:.4f}")
    
    # Keep the best k
    if score > best_score:
        best_score = score
        best_labels = labels
        best_k = k

print(f"\nBest k according to silhouette score: {best_k} with score {best_score:.4f}")

# Assign cluster labels to your truck DataFrame
df_truck_distributions['cluster_wasserstein'] = best_labels
'''
# Note that best number of clusters are manually assigned to k=2 to avoid exponential increase of cross clusters. 
# Besides, we dont really use the hard clustering of gradient because we combine it into challenge factor

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np

sil_scores = []

# Loop over k to compute silhouette scores
for k in range(2, 7):
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)
    
    # Compute silhouette score using precomputed distance
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    sil_scores.append((k, score))
    
    print(f"k={k}, silhouette score={score:.4f}")
    
    # Assign cluster labels only for k = 2
    if k == 2:
        labels_k2 = labels

# Assign the k=2 cluster labels to your truck DataFrame
df_truck_distributions['cluster_wasserstein'] = labels_k2

# Optional: print cluster counts for k=2
print("\nCluster counts for k=2:")
print(df_truck_distributions['cluster_wasserstein'].value_counts())

# 9.	CROSSTABS
import pandas as pd

# -------------------------------
# Map each VIN to its sales group
# -------------------------------
df_vin_group = df_gradient_all.select('vin', 'sales_group').distinct().toPandas()

# Align VIN order with df_truck_distributions
df_vin_group = df_vin_group.set_index('vin').loc[df_truck_distributions.index]

# Add sales_group column to truck distributions
df_truck_distributions['sales_group'] = df_vin_group['sales_group'].values

# -------------------------------
# Create cross-tab: cluster vs sales group
# -------------------------------
crosstab = pd.crosstab(df_truck_distributions['cluster_wasserstein'],
                       df_truck_distributions['sales_group'])

print(crosstab)

display(crosstab)

# 10.	PLOTS FOR VISUALIZATIONS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# Create a DataFrame from probability matrix
# -------------------------------
X_probs_df = pd.DataFrame(X_probs, index=df_truck_distributions.index, columns=bin_centers)
X_probs_df['cluster'] = df_truck_distributions['cluster_wasserstein']

# -------------------------------
# Compute average distribution per cluster
# -------------------------------
cluster_profiles = X_probs_df.groupby('cluster').mean()

print(cluster_profiles)

# Exclude 'cluster' column if present and convert bin centers to float
bin_centers_numeric = X_probs_df.columns[:-1].astype(float)

# -------------------------------
# Plot average gradient distribution per cluster
# -------------------------------
plt.figure(figsize=(10,6))

for cluster_label in cluster_profiles.index:
    plt.plot(bin_centers_numeric, cluster_profiles.loc[cluster_label, :], 
             marker='o', markersize=4, label=f'Cluster {cluster_label}')

plt.xlabel("Std Deviation Road Gradient (%)")
plt.ylabel("Normalized Mileage")
plt.title("Average gradient distribution per Wasserstein cluster")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 1. Ensure cluster & sales_group exist
# -------------------------------
assert 'cluster_wasserstein' in df_truck_distributions.columns, "Cluster column missing"
assert 'sales_group' in df_truck_distributions.columns, "Sales group column missing"

'''
# -------------------------------
# 2. Define cluster colors for 2 clusters
# -------------------------------
cluster_colors = {
    0: "#ff7f0e",  # orange (Hilly)
    1: "#1f77b4"   # blue (Flat)
}

# -------------------------------
# 3. Define custom text labels
# -------------------------------
cluster_labels = {
    0: "Hilly",
    1: "Flat"
}
'''

# -------------------------------
# 2. Define cluster colors for 2 clusters
# -------------------------------
cluster_colors = {
    0: "#1f77b4",   # blue (Hilly)
    1: "#ff7f0e"  # orange (Flat)
}

# -------------------------------
# 3. Define custom text labels
# -------------------------------
cluster_labels = {
    0: "Hilly",
    1: "Flat"
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

    # Ensure clusters in order 0,1
    counts = counts.reindex([0,1], fill_value=0)

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
    ax.set_title(f"Sales Group {sg}: Gradient Cluster Distribution", fontsize=14)

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

    # Ensure clusters [0, 1] always exist
    counts = counts.reindex([0, 1], fill_value=0)

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
df_cluster_spark.write.mode("overwrite").saveAsTable("pt_configuration_thesis.alltrucks_v1.gradient_cluster_results")


