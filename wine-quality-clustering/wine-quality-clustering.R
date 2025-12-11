# ==
# PROJECT 10: WINE QUALITY CLUSTERING USING K-MEANS
# ==

# Load required libraries
library(tidyverse)
library(cluster)
library(ggplot2)

# Load wine datasets
df1 <- read.csv("data/winequality-red.csv", sep = ";", stringsAsFactors = TRUE)
df2 <- read.csv("data/winequality-white.csv", sep = ";", stringsAsFactors = TRUE)

# Exploratory Data Analysis
cat(" RED WINE DATASET (FIRST 6 ROWS) \n")
print(head(df1))

cat("\n WHITE WINE DATASET (FIRST 6 ROWS) \n")
print(head(df2))

# Add label column to each dataset
df1$label <- sapply(df1$pH, function(x){"red"})
df2$label <- sapply(df2$pH, function(x){"white"})

# Combine datasets
wine <- rbind(df1, df2)

cat("\n COMBINED WINE DATASET \n")
cat("Dimensions:", dim(wine), "\n")
cat("Red wine samples:", sum(wine$label == "red"), "\n")
cat("White wine samples:", sum(wine$label == "white"), "\n")

# Visualizations
# Residual sugar distribution
ggplot(data = wine, aes(residual.sugar)) +
  geom_histogram(aes(fill = label), color = "black", bins = 50) +
  scale_fill_manual(values = c("#ae4554", "#faf7ea")) +
  labs(title = "Distribution of Residual Sugar by Wine Type",
       x = "Residual Sugar", y = "Count") +
  theme_minimal()

# Citric acid distribution
ggplot(data = wine, aes(citric.acid)) +
  geom_histogram(aes(fill = label), color = "black", bins = 50) +
  scale_fill_manual(values = c("#ae4554", "#faf7ea")) +
  labs(title = "Distribution of Citric Acid by Wine Type",
       x = "Citric Acid", y = "Count") +
  theme_minimal()

# Alcohol distribution
ggplot(data = wine, aes(alcohol)) +
  geom_histogram(aes(fill = label), color = "black", bins = 50) +
  scale_fill_manual(values = c("#ae4554", "#faf7ea")) +
  labs(title = "Distribution of Alcohol Content by Wine Type",
       x = "Alcohol Content", y = "Count") +
  theme_minimal()

# Scatter plot: citric acid vs residual sugar
ggplot(data = wine, aes(citric.acid, residual.sugar)) +
  geom_point(aes(colour = label), alpha = 0.2, size = 2) + 
  scale_color_manual(values = c("#ae4554", "#faf7ea")) +
  labs(title = "Citric Acid vs Residual Sugar",
       x = "Citric Acid", y = "Residual Sugar") +
  theme_dark()

# Scatter plot: volatile acidity vs residual sugar
ggplot(data = wine, aes(volatile.acidity, residual.sugar)) +
  geom_point(aes(colour = label), alpha = 0.2, size = 2) + 
  scale_color_manual(values = c("#ae4554", "#faf7ea")) +
  labs(title = "Volatile Acidity vs Residual Sugar",
       x = "Volatile Acidity", y = "Residual Sugar") +
  theme_dark()

# K-Means Clustering
# Prepare data for clustering (exclude label)
clus.data <- wine[,1:12]

cat("\n CLUSTERING DATA DIMENSIONS \n")
cat("Features:", ncol(clus.data), "\n")
cat("Samples:", nrow(clus.data), "\n")

# Apply K-Means with 2 clusters
set.seed(101)
wine.cluster <- kmeans(clus.data, 2, nstart = 20)

cat("\n K-MEANS CLUSTERING RESULTS \n")
cat("Cluster sizes:\n")
print(wine.cluster$size)

cat("\nCluster centers (first 5 features):\n")
print(wine.cluster$centers[,1:5])

# Compare clusters with actual labels
cluster_comparison <- table(wine$label, wine.cluster$cluster)
cat("\n CLUSTER VS ACTUAL LABEL COMPARISON \n")
print(cluster_comparison)

# Calculate clustering accuracy
# Assuming cluster 1 corresponds to red wine and cluster 2 to white wine
# Find which cluster has majority of red wine
red_cluster <- ifelse(sum(wine.cluster$cluster[wine$label == "red"] == 1) > 
                        sum(wine.cluster$cluster[wine$label == "red"] == 2), 1, 2)

# Calculate accuracy
if(red_cluster == 1) {
  accuracy <- (cluster_comparison[1,1] + cluster_comparison[2,2]) / sum(cluster_comparison)
} else {
  accuracy <- (cluster_comparison[1,2] + cluster_comparison[2,1]) / sum(cluster_comparison)
}

cat("\n CLUSTERING PERFORMANCE \n")
cat("Clustering Accuracy:", round(accuracy, 4), "\n")
cat("Percentage correctly clustered:", round(accuracy * 100, 2), "%\n")

# Visualize clustering results
# Create a dataframe with clustering results
wine_with_clusters <- wine
wine_with_clusters$cluster <- as.factor(wine.cluster$cluster)

# Plot alcohol vs residual sugar with clusters
ggplot(wine_with_clusters, aes(x = alcohol, y = residual.sugar, color = cluster)) +
  geom_point(alpha = 0.5, size = 2) +
  labs(title = "Wine Clusters: Alcohol vs Residual Sugar",
       x = "Alcohol Content", y = "Residual Sugar",
       color = "Cluster") +
  theme_minimal()