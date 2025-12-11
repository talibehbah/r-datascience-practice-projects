# ==
# PROJECT: BOSTON HOUSING PRICE PREDICTION USING NEURAL NETWORKS
# ==

# Load required libraries
library(MASS)
library(neuralnet)
library(caTools)
library(ggplot2)

# Load Boston housing dataset
data(Boston)

# Exploratory Data Analysis
cat(" BOSTON HOUSING DATASET OVERVIEW \n")
head(Boston)
str(Boston)
cat("\nDataset dimensions:", dim(Boston), "\n")

# Check summary statistics
cat("\n SUMMARY STATISTICS \n")
print(summary(Boston))

# Set seed for reproducibility
set.seed(101)

# Normalize the data
maxs <- apply(Boston, 2, max)
mins <- apply(Boston, 2, min)

cat("\n DATA NORMALIZATION \n")
cat("Maximum values:\n")
print(maxs)
cat("\nMinimum values:\n")
print(mins)

# Scale data to [0, 1] range
scaled.data <- scale(Boston, center = mins, scale = maxs - mins)
scaled <- as.data.frame(scaled.data)

cat("\n SCALED DATA (FIRST 6 ROWS) \n")
print(head(scaled))

# Split data
split <- sample.split(scaled$medv, SplitRatio = 0.7)
train <- subset(scaled, split == TRUE)
test <- subset(scaled, split == FALSE)

cat("\n DATA SPLIT \n")
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Prepare formula for neural network
n <- names(train)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

cat("\n NEURAL NETWORK FORMULA \n")
print(f)

# Build neural network with 2 hidden layers (5 nodes in first, 3 in second)
cat("\n BUILDING NEURAL NETWORK \n")
nn <- neuralnet(f, data = train, hidden = c(5, 3), linear.output = TRUE)

# Plot neural network
plot(nn, main = "Neural Network Architecture for Boston Housing")

# Make predictions
cat("\n MAKING PREDICTIONS \n")
predicted.nn.values <- compute(nn, test[1:13])

cat("\n PREDICTION STRUCTURE \n")
str(predicted.nn.values)

# Convert predictions back to original scale
true.predictions <- predicted.nn.values$net.result * (max(Boston$medv) - min(Boston$medv)) + min(Boston$medv)

# Convert test data back to original scale
test.r <- (test$medv) * (max(Boston$medv) - min(Boston$medv)) + min(Boston$medv)

# Calculate Mean Squared Error
MSE.nn <- sum((test.r - true.predictions)^2) / nrow(test)

cat("\n MODEL PERFORMANCE \n")
cat("Mean Squared Error (MSE):", MSE.nn, "\n")
cat("Root Mean Squared Error (RMSE):", sqrt(MSE.nn), "\n")

# Create error dataframe
error.df <- data.frame(
  Actual = test.r,
  Predicted = true.predictions,
  Error = test.r - true.predictions
)

colnames(error.df) <- c("Actual", "Predicted", "Error")

cat("\n PREDICTION RESULTS (FIRST 10 ROWS) \n")
print(head(error.df, 10))

# Calculate additional metrics
MAE <- mean(abs(error.df$Error))
R_squared <- 1 - (sum(error.df$Error^2) / sum((error.df$Actual - mean(error.df$Actual))^2))

cat("\n ADDITIONAL METRICS \n")
cat("Mean Absolute Error (MAE):", MAE, "\n")
cat("R-squared:", R_squared, "\n")

# Visualize predictions vs actual values
ggplot(error.df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Housing Prices",
       subtitle = paste("RMSE =", round(sqrt(MSE.nn), 3), "| R² =", round(R_squared, 3)),
       x = "Actual Median Value ($1000s)",
       y = "Predicted Median Value ($1000s)") +
  theme_minimal()

# Visualize error distribution
ggplot(error.df, aes(x = Error)) +
  geom_histogram(bins = 30, fill = "coral", color = "white", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Distribution of Prediction Errors",
       x = "Prediction Error (Actual - Predicted)",
       y = "Count") +
  theme_minimal()

# Compare with linear regression
cat("\n COMPARISON WITH LINEAR REGRESSION \n")
lm_model <- lm(medv ~ ., data = as.data.frame(Boston)[split == TRUE, ])
lm_predictions <- predict(lm_model, as.data.frame(Boston)[split == FALSE, ])
lm_mse <- mean((test.r - lm_predictions)^2)

cat("Neural Network MSE:", round(MSE.nn, 3), "\n")
cat("Linear Regression MSE:", round(lm_mse, 3), "\n")
cat("Improvement with Neural Network:", round(lm_mse - MSE.nn, 3), "\n")