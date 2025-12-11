# ==
# PROJECT: COLLEGE ADMISSIONS CLASSIFICATION USING DECISION TREES & RANDOM FOREST
# ==

# Load required libraries
library(tidyverse)
library(ISLR2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caTools)
library(ggplot2)

# Load College dataset
data(College)

# Exploratory Data Analysis
cat(" COLLEGE DATASET OVERVIEW \n")
head(College)
str(College)

# Visualizations
# Scatterplot of Grad.Rate versus Room.Board, colored by Private
ggplot(data = College, aes(x = Room.Board, y = Grad.Rate)) +
  geom_point(aes(colour = Private), size = 2) +
  labs(title = "Room & Board vs Graduation Rate",
       x = "Room and Board Costs", y = "Graduation Rate") +
  theme_minimal()

# Histogram of full time undergrad students, color by Private
ggplot(data = College, aes(x = F.Undergrad)) +
  geom_histogram(aes(fill = Private), colour = "black") +
  labs(title = "Distribution of Full-Time Undergraduate Students",
       x = "Number of Full-Time Undergraduates", y = "Count") +
  theme_minimal()

# Histogram of Grad.Rate colored by Private
ggplot(data = College, aes(x = Grad.Rate)) +
  geom_histogram(aes(fill = Private), colour = "black") +
  labs(title = "Distribution of Graduation Rates",
       x = "Graduation Rate (%)", y = "Count") +
  theme_minimal()

# Check for graduation rates over 100%
cat("\n GRADUATION RATES OVER 100% \n")
over_100 <- College %>% filter(Grad.Rate > 100)
print(over_100$Grad.Rate)

# Confirm their are no outliers
ggplot(data = College, aes(x = Grad.Rate)) +
  geom_histogram(aes(fill = Private), colour = "black") +
  labs(title = "Distribution of Graduation Rates",
       x = "Graduation Rate (%)", y = "Count") +
  theme_minimal()

# Fix graduation rates over 100%
College$Grad.Rate[College$Grad.Rate > 100] <- 100

cat("\n GRADUATION RATES AFTER CORRECTION \n")
cat("Maximum graduation rate now:", max(College$Grad.Rate), "\n")

# Split data
set.seed(101)
split <- sample.split(College$Private, SplitRatio = 0.7)
train <- subset(College, split == TRUE)
test <- subset(College, split == FALSE)

cat("\n DATA SPLIT \n")
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# DECISION TREE
cat("\n DECISION TREE MODEL \n")
tree <- rpart(Private ~ ., method = "class", data = train)

# Print complexity parameter
printcp(tree)

# Visualize decision tree
prp(tree, main = "Decision Tree for College Classification")

# Make predictions with decision tree
tree.preds <- predict(tree, test)

cat("\n DECISION TREE PREDICTIONS (FIRST 10) \n")
print(head(tree.preds))

# Convert probabilities to class labels
tree.preds <- as.data.frame(tree.preds)
joiner <- function(x) {
  if (x >= 0.5) {
    return('Yes')
  } else {
    return("No")
  }
}

tree.preds$Private <- sapply(tree.preds$Yes, joiner)

cat("\n DECISION TREE PREDICTIONS WITH CLASS LABELS (FIRST 10) \n")
print(head(tree.preds))

# Decision Tree Confusion Matrix
cat("\n DECISION TREE CONFUSION MATRIX \n")
print(table(tree.preds$Private, test$Private))

# RANDOM FOREST
cat("\n RANDOM FOREST MODEL \n")
rf.model <- randomForest(Private ~ ., data = train, importance = TRUE)

# Random Forest Summary
print(rf.model)

# Variable Importance
cat("\n RANDOM FOREST VARIABLE IMPORTANCE \n")
print(rf.model$importance)

# Make predictions with Random Forest
rf.preds <- predict(rf.model, test)

# Random Forest Confusion Matrix
cat("\n RANDOM FOREST CONFUSION MATRIX \n")
print(table(rf.preds, test$Private))

# Compare model performance
tree.accuracy <- mean(tree.preds$Private == test$Private)
rf.accuracy <- mean(rf.preds == test$Private)

cat("\n MODEL COMPARISON \n")
cat("Decision Tree Accuracy:", round(tree.accuracy, 4), "\n")
cat("Random Forest Accuracy:", round(rf.accuracy, 4), "\n")
cat("Improvement with Random Forest:", round(rf.accuracy - tree.accuracy, 4), "\n")


