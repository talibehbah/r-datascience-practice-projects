# =============================================================================
# PROJECT: EMPLOYEE ATTRITION ANALYSIS USING LOGISTIC REGRESSION
# =============================================================================

# Load required libraries
library(tidyverse)
library(caret)
library(caTools)
library(corrplot)
library(Amelia)

# Set seed for reproducibility
set.seed(101)

# Load data
emp_attr <- read.csv("data/Employee-Attrition.csv", stringsAsFactors = TRUE)

# Data preprocessing
# Convert categorical variables to factors
factor_vars <- c("BusinessTravel", "Department", "EducationField", "Gender",
                 "JobRole", "MaritalStatus", "OverTime")
emp_attr[factor_vars] <- lapply(emp_attr[factor_vars], factor)

# Convert ordinal numeric variables to factors
ordinal_vars <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                  "JobLevel", "JobSatisfaction", "RelationshipSatisfaction",
                  "WorkLifeBalance", "StockOptionLevel")
emp_attr[ordinal_vars] <- lapply(emp_attr[ordinal_vars], factor)

# Remove useless columns
emp_attr <- emp_attr %>%
  select(-c(EmployeeCount, EmployeeNumber, StandardHours, Over18, MonthlyRate, DailyRate))

# Feature engineering
emp_attr$TenureRatio <- emp_attr$YearsAtCompany / emp_attr$TotalWorkingYears
emp_attr$HighOverTime <- ifelse(emp_attr$OverTime == "Yes", 1, 0)

# Exploratory Data Analysis
cat("=== DATA STRUCTURE ===\n")
str(emp_attr)

# Visualizations
ggplot(emp_attr, aes(Age)) + 
  geom_histogram(aes(fill = Attrition), color = 'black', binwidth = 1) + 
  theme_bw()

# Attrition counts
ggplot(emp_attr, aes(x = Attrition, fill = Attrition)) +
  geom_bar() +
  labs(title = "Attrition Count", y = "Number of Employees") +
  theme_minimal()

# Attrition by Department
ggplot(emp_attr, aes(x = Department, fill = Attrition)) +
  geom_bar(position = "dodge") +
  labs(title = "Attrition by Department", y = "Count") +
  theme_minimal()

# Attrition by JobRole
ggplot(emp_attr, aes(x = JobRole, fill = Attrition)) +
  geom_bar(position = "dodge") +
  labs(title = "Attrition by Job Role", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_minimal()

# Age distribution by Attrition
ggplot(emp_attr, aes(x = Age, fill = Attrition)) +
  geom_histogram(bins = 20, alpha = 0.7, position = "identity") +
  labs(title = "Age Distribution by Attrition", x = "Age", y = "Count") +
  theme_minimal()

# Monthly Income distribution by Attrition
ggplot(emp_attr, aes(x = MonthlyIncome, fill = Attrition)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Monthly Income Distribution by Attrition", 
       x = "Monthly Income", y = "Count") +
  theme_minimal()

# Boxplot: Age vs Attrition
ggplot(emp_attr, aes(x = Attrition, y = Age, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Boxplot of Age by Attrition", y = "Age") +
  theme_minimal()

# Boxplot: Monthly Income vs Attrition
ggplot(emp_attr, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot() +
  labs(title = "Boxplot of Monthly Income by Attrition", y = "Monthly Income") +
  theme_minimal()

# Correlation heatmap for numeric variables
numeric_vars <- emp_attr %>%
  select(Age, DistanceFromHome, HourlyRate, MonthlyIncome, NumCompaniesWorked,
         PercentSalaryHike, PerformanceRating, TotalWorkingYears,
         TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole,
         YearsSinceLastPromotion, YearsWithCurrManager, TenureRatio, HighOverTime)

cor_matrix <- cor(numeric_vars, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.8, addCoef.col = "black",
         title = "Correlation Heatmap of Employee Attributes")

# Split data
set.seed(101)
split <- sample.split(emp_attr$Attrition, SplitRatio = 0.7)
train <- subset(emp_attr, split == TRUE)
test <- subset(emp_attr, split == FALSE)

cat("\n=== DATA SPLIT ===\n")
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Remove HighOverTime from train and test
train$HighOverTime <- NULL
test$HighOverTime <- NULL

# Ensure factors in test match train
for(col in names(train)) {
  if(is.factor(train[[col]])) {
    test[[col]] <- factor(test[[col]], levels = levels(train[[col]]))
  }
}

# Fit logistic regression model
attrition_model <- glm(Attrition ~ ., data = train, family = binomial)

cat("\n=== MODEL SUMMARY ===\n")
print(summary(attrition_model))

# Make predictions on test data
test_clean <- test[!is.na(test$Attrition), ]
test_prob <- predict(attrition_model, newdata = test_clean, type = "response")
test_pred <- ifelse(test_prob > 0.5, "Yes", "No")
test_pred <- factor(test_pred, levels = c("No", "Yes"))

# Calculate accuracy
accuracy <- mean(test_clean$Attrition == test_pred)
cat("\n=== MODEL ACCURACY ===\n")
cat("Test Set Accuracy:", round(accuracy, 4), "\n")

# Confusion matrix
cat("\n=== CONFUSION MATRIX ===\n")
conf_matrix <- table(Actual = test$Attrition, Predicted = test_pred)
print(conf_matrix)
