'''
0. README
Notebook: CrossTabs_Visualizer
Inputs
•	pt_configuration_thesis.alltrucks_v1.velocity_cluster_results
•	pt_configuration_thesis.alltrucks_v1.weight_cluster_results
•	pt_configuration_thesis.alltrucks_v1.gradient_cluster_results
•	pt_configuration_thesis.alltrucks_v1.sgX_talpy_stats ((X = 1,2,....15))
Outputs
•	pt_configuration_thesis.alltrucks_v1.vin_info_SG_info_Cluster_info_TALPY_info
Purpose
•	This notebook is meant to take as input the cluster results of velocity, weight and gradient from previous notebooks along with the TALPY stats generated per vin also from the previous notebook.
•	The notebook then assigns for every cluster - textual context. For example Velocity Cluster 1 = High Velocity.
•	The notebook then combines the TALPY stats together with the cluster info and writes this dataframe to the catalog.
•	Finally the notebook has a section to create crosstab table style visualizations to analyze each cluster combination for each sales group.
Other remarks
•	Use the drop down utlity widget on the top to select the sales group of interest. The table in section - 5 updates automatically in a few seconds.
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
df_velocity_cluster = spark.table("pt_configuration_thesis.alltrucks_v1.`velocity_cluster_results`")

unique_vins_count_ori = df_velocity_cluster.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

df_weight_cluster = spark.table("pt_configuration_thesis.alltrucks_v1.`weight_cluster_results`")

unique_vins_count_ori = df_weight_cluster.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

df_gradient_cluster = spark.table("pt_configuration_thesis.alltrucks_v1.`gradient_cluster_results`")

unique_vins_count_ori = df_gradient_cluster.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

dfs = []

# Loop over sales groups 1 to 15
for sg in range(1, 16):
    table_name = f"pt_configuration_thesis.alltrucks_v1.sg{sg}_talpy_stats"
    
    # Load Spark table
    df = spark.table(table_name)
    
    # Add sales_group column
    df = df.withColumn("sales_group", F.lit(sg))
    
    dfs.append(df)

# Union all DataFrames into a single DataFrame
df_all_groups = reduce(lambda df1, df2: df1.unionByName(df2), dfs)

# Show sample
df_all_groups.show(10)

display(df_all_groups)

unique_vins_count_ori = df_all_groups.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	PREPARE CLUSTERING RESULTS DATA
from pyspark.sql.functions import col, when

# 1. Rename the cluster column
df_velocity_cluster = df_velocity_cluster.withColumnRenamed(
    "cluster_wasserstein", "cluster_wasserstein_velocity"
)

# 2. Add the text column based on cluster values
# Note that I am currently doing this manually by looking at the plots in the notebook Wasserstein_Distance_Velocity_Spectra
df_velocity_cluster = df_velocity_cluster.withColumn(
    "cluster_velocity_text",
    when(col("cluster_wasserstein_velocity") == 1, "High")
    .when(col("cluster_wasserstein_velocity") == 2, "Low")
    .when(col("cluster_wasserstein_velocity") == 0, "Medium")
)

# Show sample
df_velocity_cluster.show(10)

display(df_velocity_cluster)

from pyspark.sql.functions import col, when

# 1. Rename the cluster column
df_weight_cluster = df_weight_cluster.withColumnRenamed(
    "cluster_wasserstein", "cluster_wasserstein_weight"
)

# 2. Add the text column based on cluster values
# Note that I am currently doing this manually by looking at the plots in the notebook Wasserstein_Distance_Weight_Spectra
df_weight_cluster = df_weight_cluster.withColumn(
    "cluster_weight_text",
    when(col("cluster_wasserstein_weight") == 0, "High")
    .when(col("cluster_wasserstein_weight") == 1, "Low")
)

# Show sample
df_weight_cluster.show(10)

display(df_weight_cluster)

from pyspark.sql.functions import col, when

# 1. Rename the cluster column
df_gradient_cluster = df_gradient_cluster.withColumnRenamed(
    "cluster_wasserstein", "cluster_wasserstein_gradient"
)

# 2. Add the text column based on cluster values
# Note that I am currently doing this manually by looking at the plots in the notebook Wasserstein_Distance_Gradient_Spectra
df_gradient_cluster = df_gradient_cluster.withColumn(
    "cluster_gradient_text",
    when(col("cluster_wasserstein_gradient") == 1, "Flat")
    .when(col("cluster_wasserstein_gradient") == 0, "Hilly")
)

# Show sample
df_gradient_cluster.show(10)

display(df_gradient_cluster)

# Join velocity and weight clusters 

df_combined_cluster = df_velocity_cluster.join(
    df_weight_cluster,
    on=["vin", "sales_group"],
    how="inner"
)

# Then join with gradient clusters
df_combined_cluster = df_combined_cluster.join(
    df_gradient_cluster,
    on=["vin", "sales_group"],
    how="inner"
)

# Show sample
df_combined_cluster.show(10)

display(df_combined_cluster)

# Total number of unique VINs
total_vins = df_combined_cluster.select("vin").distinct().count()
print(f"Total number of unique VINs: {total_vins}")

# Number of VINs per sales group
vins_per_sales_group = df_combined_cluster.groupBy("sales_group").agg({"vin": "count"}).withColumnRenamed("count(vin)", "num_vins")
vins_per_sales_group.show()

# 4.	COMBINE TALPY STATS WITH CLUSTERING RESULTS AND WRITE DATA TO CATALOG
df_cluster_talpy_combined = df_combined_cluster.join(
    df_all_groups,
    on=["vin", "sales_group"],
    how="inner"
)


# Check total number of VINs to confirm
total_vins = df_cluster_talpy_combined.select("vin").distinct().count()
print(f"Total VINs in combined cluster+talpy DataFrame: {total_vins}")

# Number of VINs per sales group
vins_per_sales_group = df_cluster_talpy_combined.groupBy("sales_group").agg({"vin": "count"}).withColumnRenamed("count(vin)", "num_vins")
vins_per_sales_group.show()

display(df_cluster_talpy_combined)

df_cluster_talpy_combined.write.mode("overwrite").saveAsTable("pt_configuration_thesis.alltrucks_v1.vin_info_SG_info_Cluster_info_TALPY_info")

# 5.	CROSSTABS VISUALIZER
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.gridspec as gridspec
from pyspark.sql import functions as F


# Section 1 - Dropdown for Sales Group
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("sales_group", "1", [str(i) for i in range(1, 16)], "Select Sales Group")
selected_sales_group = int(dbutils.widgets.get("sales_group"))
print(f"Selected Sales Group: {selected_sales_group}")


# Section 2 - Filter DataFrame based on what is selected 
df_sg = df_cluster_talpy_combined.filter(df_cluster_talpy_combined.sales_group == selected_sales_group)
total_vins = df_sg.select("vin").distinct().count()
print(f"Total VINs in Sales Group {selected_sales_group}: {total_vins}")
df_sg_pd = df_sg.toPandas()


# Section 3 - Define clusters & what they mean (check with excel in drive or notebooks responsible for clustering!!)

velocity_clusters = [1, 0, 2]  # High, Medium, Low
velocity_labels = {0: "Medium", 1: "High", 2: "Low"}

weight_clusters = [0, 1]  # High Weight, Low Weight
weight_labels = {0: "High Weight", 1: "Low Weight"}

gradient_clusters = [0, 1]  # Hilly, Flat
gradient_labels = {0: "Hilly Road", 1: "Flat Road"}

columns = [
    f"{gradient_labels[g]}\n{weight_labels[w]}"
    for g, w in product(gradient_clusters, weight_clusters)
]

rows = [velocity_labels[v] for v in velocity_clusters]


# Section 4 - Initialize empty matrices
metrics = {
    "Number of Trucks (-)": pd.DataFrame(0, index=rows, columns=columns, dtype=float),
    "Avg Driving Velocity (kmph)": pd.DataFrame(float("nan"), index=rows, columns=columns),
    "Avg GCW (kg)": pd.DataFrame(float("nan"), index=rows, columns=columns),
    "Std Dev Road Gradient (%)": pd.DataFrame(float("nan"), index=rows, columns=columns),
    "Fuel Consumption (l/100km)": pd.DataFrame(float("nan"), index=rows, columns=columns)
}


# Section 5 - opulate Metrics

for v, g, w in product(velocity_clusters, gradient_clusters, weight_clusters):

    subset = df_sg_pd[
        (df_sg_pd["cluster_wasserstein_velocity"] == v) &
        (df_sg_pd["cluster_wasserstein_weight"] == w) &
        (df_sg_pd["cluster_wasserstein_gradient"] == g)
    ]

    row_label = velocity_labels[v]
    col_label = f"{gradient_labels[g]}\n{weight_labels[w]}"

    if subset.shape[0] == 0:
        metrics["Number of Trucks"].loc[row_label, col_label] = 0
        continue

    metrics["Number of Trucks (-)"].loc[row_label, col_label] = subset.shape[0]
    metrics["Avg Driving Velocity (kmph)"].loc[row_label, col_label] = subset["vehicle_speed_avg_driving_kmh"].mean()
    metrics["Avg GCW (kg)"].loc[row_label, col_label] = subset["gc_weight_avg_kg"].mean()
    metrics["Std Dev Road Gradient (%)"].loc[row_label, col_label] = subset["stddev_road_gradient"].mean()
    metrics["Fuel Consumption (l/100km)"].loc[row_label, col_label] = subset["fuel_cons_driving_l_per_100km"].mean()


# Section 6 - Plot heatmaps - 2 Rows and 3 Columns

fig = plt.figure(figsize=(26, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
metric_names = list(metrics.keys())

for idx, metric_name in enumerate(metric_names):
    df_metric = metrics[metric_name]

    # First 3 heatmaps in row 1, next 2 in row 2
    row = 0 if idx < 3 else 1
    col = idx if idx < 3 else idx - 3

    ax = fig.add_subplot(gs[row, col])

    sns.heatmap(
        df_metric,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        square=True,
        ax=ax,
        mask=df_metric.isna(),
        annot_kws={"fontsize": 18}
    )

    ax.set_title(f"{metric_name}", fontsize=18, fontweight='bold')
    ax.set_ylabel("Velocity Cluster", fontsize=14)
    ax.set_xlabel("")

    # Labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha='center',
        fontsize=14,
        linespacing=1.5            # space between line 1 & line 2
    )

    # Y-axis label font adjustments
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=14
    )

plt.suptitle(f"Cluster Summary CrossTabs - Sales Group {selected_sales_group}", fontsize=26, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
