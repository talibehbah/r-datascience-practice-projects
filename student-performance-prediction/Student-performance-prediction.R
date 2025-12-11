# =============================================================================
# PROJECT: STUDENT PERFORMANCE PREDICTION USING LINEAR REGRESSION
# =============================================================================

# Load required libraries
library(tidyverse)
library(caret)
library(corrplot)
library(caTools)

# Set seed for reproducibility
set.seed(101)

# Load data
student <- read.csv("data/student-mat.csv", sep = ";", stringsAsFactors = TRUE)

# Exploratory Data Analysis
head(student)
str(student)

# Check for missing values
any(is.na(student))

# Correlation Analysis
num.col <- sapply(student, is.numeric)
cor.data <- cor(student[, num.col])
corrplot(corr = cor.data, method = "color")

# Visualize target variable
ggplot(data = student, aes(x = G3)) +
  geom_histogram(bins = 20, fill = "blue", color = "red") +
  theme_minimal()

# Split data into training and testing sets
sample <- sample.split(student$G3, SplitRatio = 0.7)
student_train <- subset(student, sample == TRUE)
student_test <- subset(student, sample == FALSE)

# Build linear regression model
student_model <- lm(G3 ~ ., data = student_train)

# Model summary
summary(student_model)

# Residual analysis
student_res <- residuals(student_model)
student_res <- as.data.frame(student_res)

ggplot(student_res, aes(student_res)) +
  geom_histogram(fill = "lightblue", color = "darkgreen") +
  theme_minimal()

# Make predictions
G3.predictions <- predict(student_model, student_test)

# Create results dataframe
G3.predictions_results <- cbind(G3.predictions, student_test$G3)
colnames(G3.predictions_results) <- c("predicted", "actual")
G3.predictions_results <- as.data.frame(G3.predictions_results)
head(G3.predictions_results)

# Handle negative predictions
to_zero <- function(x) {
  if (x < 0) {
    return(0)
  } else {
    return(x)
  }
}

G3.predictions_results$predicted <- sapply(G3.predictions_results$predicted, to_zero)

# Calculate evaluation metrics
MSE <- mean((G3.predictions_results$actual - G3.predictions_results$predicted)^2)
RMSE <- sqrt(MSE)

SSE <- sum((G3.predictions_results$predicted - G3.predictions_results$actual)^2)
SST <- sum((mean(student$G3) - G3.predictions_results$actual)^2)
R2 <- 1 - (SSE / SST)

# Print model performance
cat("\n=== MODEL PERFORMANCE ===\n")
cat("R-squared:", round(R2, 3), "\n")
cat("RMSE:", round(RMSE, 3), "\n")
cat("MSE:", round(MSE, 3), "\n")