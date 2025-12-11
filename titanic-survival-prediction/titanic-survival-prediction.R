# =============================================================================
# PROJECT: TITANIC SURVIVAL PREDICTION USING LOGISTIC REGRESSION
# =============================================================================

# Load required libraries
library(tidyverse)
library(Amelia)
library(caTools)

# Load data
titanic_train <- read.csv("data/titanic_train.csv", stringsAsFactors = TRUE)
titanic_test <- read.csv("data/titanic_test.csv", stringsAsFactors = TRUE)

# Exploratory Data Analysis
head(titanic_train)
str(titanic_train)

# Check for missing values
missmap(titanic_train, main = "Missing Map", col = c("red", "green"), legend = FALSE)

# Visualizations
ggplot(data = titanic_train, aes(Survived)) +
  geom_bar(fill = "darkblue") +
  theme_minimal()

ggplot(data = titanic_train, aes(Survived)) +
  geom_bar(aes(fill = factor(Pclass))) +
  theme_minimal()

ggplot(data = titanic_train, aes(Survived)) +
  geom_bar(aes(fill = factor(Sex))) +
  theme_minimal()

ggplot(data = titanic_train, aes(Age)) +
  geom_histogram(fill = "lightblue", color = "darkred") +
  theme_minimal()

ggplot(data = titanic_train, aes(SibSp)) +
  geom_bar(fill = "darkblue") +
  theme_minimal()

ggplot(data = titanic_train, aes(Fare)) +
  geom_histogram(fill = "lightblue", color = "darkred") +
  theme_minimal()

ggplot(data = titanic_train, aes(Pclass, Age)) +
  geom_boxplot(aes(group = Pclass, fill = factor(Pclass), alpha = 0.4)) +
  scale_y_continuous(breaks = seq(min(0), max(80), by = 2)) +
  theme_minimal()

# Impute missing age values based on passenger class
impute_age <- function(age, class) {
  out <- age
  for (i in 1:length(age)) {
    if (is.na(age[i])) {
      if (class[i] == 1) {
        out[i] <- 37
      } else if (class[i] == 2) {
        out[i] <- 29
      } else {
        out[i] <- 24
      }
    } else {
      out[i] <- age[i]
    }
  }
  return(out)
}


# Confirm missing values are now fixed
missmap(titanic_train, main = "Missing Map", col = c("red", "green"), legend = FALSE)

# Apply imputation to training data
fixed.ages <- impute_age(titanic_train$Age, titanic_train$Pclass)
titanic_train$Age <- fixed.ages

# Prepare training data
titanic_train <- titanic_train %>% 
  select(-PassengerId, -Name, -Ticket, -Cabin)

# Convert to factors
titanic_train$Survived <- factor(titanic_train$Survived)
titanic_train$Pclass <- factor(titanic_train$Pclass)
titanic_train$Parch <- factor(titanic_train$Parch)
titanic_train$SibSp <- factor(titanic_train$SibSp)

str(titanic_train)

# Build logistic regression model
titanic_model <- glm(Survived ~ ., family = binomial(link = "logit"), 
                     data = titanic_train)

# Model summary
summary(titanic_model)

# Prepare test data
head(titanic_test)
str(titanic_test)

# Check missing values in test data
missmap(titanic_test, main = "Missing Map", col = c("red", "green"), legend = FALSE)

# Impute age for test data
fixed.ages_test <- impute_age(titanic_test$Age, titanic_test$Pclass)
titanic_test$Age <- fixed.ages_test

titanic_test <- titanic_test %>% 
  select(-PassengerId, -Name, -Ticket, -Cabin)

# Handle missing Fare in test data
titanic_test$Fare[is.na(titanic_test$Fare)] <- mean(titanic_test$Fare, na.rm = TRUE)

# Fix outlier in Parch
titanic_test$Parch[titanic_test$Parch == 9] <- 6

# Convert to factors
titanic_test$Pclass <- factor(titanic_test$Pclass)
titanic_test$Parch <- factor(titanic_test$Parch)
titanic_test$SibSp <- factor(titanic_test$SibSp)

str(titanic_test)

# Make predictions on training data
train_prob <- predict(titanic_model, newdata = titanic_train, type = "response")
train_pred <- ifelse(train_prob > 0.5, 1, 0)

# Confusion matrix for training data
cat("\n=== TRAINING DATA CONFUSION MATRIX ===\n")
print(table(Actual = titanic_train$Survived, Predicted = train_pred))

# Make predictions on test data
test_prob <- predict(titanic_model, newdata = titanic_test, type = "response")
titanic_test$Survived <- ifelse(test_prob > 0.5, 1, 0)

# View test predictions
cat("\n=== TEST DATA PREDICTIONS (FIRST 10 ROWS) ===\n")
print(head(titanic_test, 10))