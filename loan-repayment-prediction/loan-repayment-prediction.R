# ==
# PROJECT: LOAN REPAYMENT PREDICTION USING SUPPORT VECTOR MACHINES
# ==

# Load required libraries
library(e1071)
library(caTools)
library(ggplot2)

# Load loan data
loan <- read.csv("data/loan_data.csv", stringsAsFactors = TRUE)

# Exploratory Data Analysis
cat(" LOAN DATA OVERVIEW \n")
str(loan)
summary(loan)

# Convert variables to factors
loan$inq.last.6mths <- factor(loan$inq.last.6mths)
loan$delinq.2yrs <- factor(loan$delinq.2yrs)
loan$pub.rec <- factor(loan$pub.rec)
loan$not.fully.paid <- factor(loan$not.fully.paid)
loan$credit.policy <- factor(loan$credit.policy)

cat("\n DATA STRUCTURE AFTER CONVERSION \n")
str(loan)

# Visualizations
# Histogram of fico scores colored by not.fully.paid
ggplot(data = loan, aes(x = fico)) +
  geom_histogram(aes(fill = not.fully.paid), color = "black") +
  scale_fill_manual(values = c('green', 'red')) +
  labs(title = "FICO Score Distribution by Loan Repayment Status",
       x = "FICO Score", y = "Count") +
  theme_minimal()

# Barplot of purpose counts, colored by not.fully.paid
ggplot(data = loan, aes(x = factor(purpose))) +
  geom_bar(aes(fill = not.fully.paid), position = "dodge") +
  labs(title = "Loan Purpose by Repayment Status",
       x = "Loan Purpose", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Scatterplot of fico score versus int.rate
ggplot(data = loan, aes(x = int.rate, y = fico)) +
  geom_point(aes(colour = not.fully.paid), size = 3, alpha = 0.5) +
  labs(title = "Interest Rate vs FICO Score by Repayment Status",
       x = "Interest Rate", y = "FICO Score") +
  theme_minimal()

# Split data
set.seed(101)
spl = sample.split(loan$not.fully.paid, 0.7)
train = subset(loan, spl == TRUE)
test = subset(loan, spl == FALSE)

cat("\n DATA SPLIT \n")
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Build SVM model with radial kernel
model <- svm(not.fully.paid ~ ., data = train)

cat("\n SVM MODEL SUMMARY \n")
print(summary(model))

# Make predictions
predicted.values <- predict(model, test[1:13])

# Confusion Matrix
cat("\n SVM CONFUSION MATRIX \n")
print(table(predicted.values, test$not.fully.paid))

# Calculate accuracy
accuracy <- mean(predicted.values == test$not.fully.paid)
cat("\nSVM Model Accuracy:", round(accuracy, 4), "\n")

# Tune SVM parameters
cat("\n TUNING SVM PARAMETERS \n")
tune.results <- tune(svm, train.x = not.fully.paid ~ ., data = train, 
                     kernel = 'radial', 
                     ranges = list(cost = c(1, 10), gamma = c(0.1, 1)))

print(summary(tune.results))

# Build tuned SVM model
tuned.model <- svm(not.fully.paid ~ ., data = train, cost = 10, gamma = 0.1)

cat("\n TUNED SVM MODEL SUMMARY \n")
print(summary(tuned.model))

# Make predictions with tuned model
tuned.predictions <- predict(tuned.model, test[1:13])

# Tuned model confusion matrix
cat("\n TUNED SVM CONFUSION MATRIX \n")
print(table(tuned.predictions, test$not.fully.paid))

# Calculate tuned model accuracy
tuned.accuracy <- mean(tuned.predictions == test$not.fully.paid)
cat("\nTuned SVM Model Accuracy:", round(tuned.accuracy, 4), "\n")
cat("Improvement from tuning:", round(tuned.accuracy - accuracy, 4), "\n")