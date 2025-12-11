# =============================================================================
# PROJECT: ADULT INCOME CLASSIFICATION USING LOGISTIC REGRESSION
# =============================================================================

# Load required libraries
library(tidyverse)
library(caTools)
library(Amelia)

# Load data
adult <- read.csv("data/adult_sal.csv", stringsAsFactors = TRUE)

# Exploratory Data Analysis
head(adult)
adult <- adult %>% select(-X)
str(adult)
summary(adult)

# Check employer type distribution
cat("=== EMPLOYER TYPE DISTRIBUTION ===\n")
print(table(adult$type_employer))

# Clean employer type - combine unemployed categories
un_emp <- function(job) {
  job <- as.character(job)
  if (job == "Never-Worked" | job == "Without-pay") {
    return("unemployed")
  } else {
    return(job)
  }
}

adult$type_employer <- sapply(adult$type_employer, un_emp)

# Group employer types
group_emp <- function(job) {
  if (job == 'Local-gov' | job == 'State-gov') {
    return('SL-gov')
  } else if (job == 'Self-emp-inc' | job == 'Self-emp-not-inc') {
    return('self-emp')
  } else {
    return(job)
  }
}

adult$type_employer <- sapply(adult$type_employer, group_emp)

cat("\n=== EMPLOYER TYPE AFTER GROUPING ===\n")
print(table(adult$type_employer))

# Clean marital status
cat("\n=== MARITAL STATUS DISTRIBUTION ===\n")
print(table(adult$marital))

group_marital <- function(mar) {
  mar <- as.character(mar)
  
  if (mar == 'Separated' | mar == 'Divorced' | mar == 'Widowed') {
    return('Not-Married')
  } else if (mar == 'Never-married') {
    return(mar)
  } else {
    return('Married')
  }
}

adult$marital <- sapply(adult$marital, group_marital)

cat("\n=== MARITAL STATUS AFTER GROUPING ===\n")
print(table(adult$marital))

# Group countries
cat("\n=== COUNTRY DISTRIBUTION ===\n")
print(table(adult$country))

# Define country groups
Asia <- c('China', 'Hong', 'India', 'Iran', 'Cambodia', 'Japan', 'Laos',
          'Philippines', 'Vietnam', 'Taiwan', 'Thailand')

North.America <- c('Canada', 'United-States', 'Puerto-Rico')

Europe <- c('England', 'France', 'Germany', 'Greece', 'Holand-Netherlands', 'Hungary',
            'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland', 'Yugoslavia')

Latin.and.South.America <- c('Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',
                             'El-Salvador', 'Guatemala', 'Haiti', 'Honduras',
                             'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru',
                             'Jamaica', 'Trinadad&Tobago')

Other <- c('South')

group_country <- function(ctry) {
  if (ctry %in% Asia) {
    return('Asia')
  } else if (ctry %in% North.America) {
    return('North.America')
  } else if (ctry %in% Europe) {
    return('Europe')
  } else if (ctry %in% Latin.and.South.America) {
    return('Latin.and.South.America')
  } else {
    return('Other')
  }
}

adult$country <- sapply(adult$country, group_country)

cat("\n=== COUNTRY AFTER GROUPING ===\n")
print(table(adult$country))

# Convert to factors
adult$type_employer <- sapply(adult$type_employer, factor)
adult$country <- sapply(adult$country, factor)
adult$marital <- sapply(adult$marital, factor)

# Handle missing values (represented as '?')
adult[adult == '?'] <- NA

cat("\n=== EMPLOYER TYPE AFTER HANDLING MISSING ===\n")
print(table(adult$type_employer))

# Convert remaining columns to factors
adult$type_employer <- sapply(adult$type_employer, factor)
adult$country <- sapply(adult$country, factor)
adult$marital <- sapply(adult$marital, factor)
adult$occupation <- sapply(adult$occupation, factor)

# Check missing values
missmap(adult, y.at = c(1), y.labels = c(''), col = c('yellow', 'black'))

# Remove NA values
adult <- na.omit(adult)

# Confirm missing values are now fixed
missmap(adult, y.at = c(1), y.labels = c(''), col = c('yellow', 'black'))


cat("\n=== DATA STRUCTURE AFTER CLEANING ===\n")
str(adult)

# Visualizations
ggplot(adult, aes(age)) + 
  geom_histogram(aes(fill = income), color = 'black', binwidth = 1) + 
  theme_bw()

ggplot(adult, aes(hr_per_week)) + 
  geom_histogram(bins = 30, fill = "lightblue", color = "red") + 
  theme_bw()

# Rename country to region
names(adult)[names(adult) == "country"] <- "region"

ggplot(adult, aes(region)) + 
  geom_bar(aes(fill = income), color = 'black') + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Split data
set.seed(101)
sample <- sample.split(adult$income, SplitRatio = 0.70)
train = subset(adult, sample == TRUE)
test = subset(adult, sample == FALSE)

cat("\n=== DATA SPLIT ===\n")
cat("Training set size:", nrow(train), "\n")
cat("Testing set size:", nrow(test), "\n")

# Build logistic regression model
model = glm(income ~ ., family = binomial(logit), data = train)

cat("\n=== MODEL SUMMARY ===\n")
print(summary(model))