#------------------------------------------
## STAT 642
## Deep Learning
#------------------------------------------
#------------------------------------------
######### Preliminary Code #########
#------------------------------------------
## Clear your workspace
# If your workspace is NOT empty, run:
rm(list=ls())
# to clear it
#------------------------------------------
## Set wd
setwd("C:/Users/chh35/OneDrive - Drexel University/Teaching/Drexel/STAT 642/Course Content/Week 10")
#------------------------------------------

## Install new packages
install.packages("keras", 
                 "tensorflow", 
                 "tfruns")

#------------------------------------------
## Load libraries
library(caret) 
library(tensorflow)
library(keras)
library(tfruns)

## Load Data
OJ <- read.csv(file = "OJ.csv",
               stringsAsFactors = FALSE)

#------------------------------------------

## Data Overview

# The OJ dataset contains purchase information
# for orange juice purchases made at 5 different
# stores. Customer and product information is
# included, and a large grocery store chain wants
# to use the data to be able to predict if a 
# customer will purchase Citrus Hill (CH) or 
# Minute Maid (MM) orange juice. The grocery
# store prioritizes being able to correctly predict
# Citrus Hill purchases.

# Variable Descriptions:
# Purchase: OJ purchase choice, either Citrus
#           Hill (CH) or Minute Maid (MM)
# Week of Purchase: number identifying the week
#                   of purchase
# Store ID: the identification of the store of
#           OJ purchase 
# PriceCH: The price charged for CH
# PriceMM: The price charged for MM
# DiscCH: Discount offered by CH
# DiscMM: Discount offered by MM
# SpecialCH: Indicates if there was a promotion
#            for CH (1) or not (0)
# SpecialMM: Indicates if there was a promotion
#            for MM (1) or not (0)
# LoyalCH: customer brand loyalty to CH
# SalePriceMM: The sale price of MM
# SalePriceCH: The sale price of CH
# PriceDiff: MM Sale Price - CH Sale Price
# PctDiscMM: Percentage discount for MM
# PctDiscCH: Percentage discount for CH
# ListPriceDiff: MM List Price - CH List Price

#------------------------------------------

## Data Exploration & Preparation

# First, we can view high-level information
# about the dataframe
str(OJ)

## Prepare Target (Y) Variable
# To make compatible with tf/keras, convert
# to binary (0 = MM, 1 = CH)
OJ$Purchase <- ifelse(OJ$Purchase == "MM",
                      yes = 0, 
                      no = 1)

# Let's visualize the distribution to check
# for class imbalance
plot(factor(OJ$Purchase), main = "Purchase")

## Prepare Predictor (X) Variables

## Categorical
# Nominal (Unordered) Factor Variables
# Our nominal variables are StoreID, SpecialCH,
# SpecialMM. SpecialCH and SpecialMM are already
# binary. we convert StoreID to a nominal factor 
# (for now) and will binarize it later.
OJ$StoreID <- factor(OJ$StoreID)

# Ordinal (Ordered) Factor Variables
# We do not have any ordinal variables

## Numeric
# All other variables are numeric

# For convenience, we can set up a vector
# of potential predictor variable names
vars <- names(OJ)[!names(OJ) %in% "Purchase"]

# We can obtain summary information for
# our prepared data
summary(OJ)

#------------------------------------------

## Data Preprocessing & Transformation

# ANN can handle redundant and irrelevant 
# variables when not in large numbers, 
# missing values need to be handled, categorical 
# variables need to be binarized and rescaling 
# should be done

## 1. Missing values
any(is.na(OJ))

## 2. Redundant Variables
# We have reason to believe that we may have 
# many highly correlated variables. 

# First, we obtain the correlation matrix
# for our numeric predictor variables
cor_vars <- cor(x = OJ[ ,!names(OJ) %in% c("Purchase", "StoreID", "SpecialCH", "SpecialMM")])

# We can use the findCorrelation() function 
# in the caret package to identify redundant 
# variables for us. 
high_corrs <- findCorrelation(x = cor_vars, 
                              cutoff = .75, 
                              names = TRUE)
high_corrs # view variable names for removal

# Now, we can remove them from our vars
# vector so that we exclude them from our
# list of input (X) variable names
vars <- vars[!vars %in% high_corrs]
vars

## 3. Binarize Categorical Variables
# We need to binarize the StoreID variable.
cats <- dummyVars(formula =  ~ StoreID,
                  data = OJ)
cats_dums <- predict(object = cats, 
                     newdata = OJ)

# Combine binarized variables (cats_dum) with data
# (OJ), excluding the StoreID factor variable
OJ_dum <- data.frame(OJ[ ,!names(OJ) %in% "StoreID"],
                     cats_dums)

# We can update our variable list
vars <- c(vars[-2], colnames(cats_dums))
vars

## 4. Rescale Numeric Variables
# We will apply min-max normalization
mmnorm <- preProcess(x = OJ_dum[,vars],
                     method = "range")
OJ_dum_mm <- predict(object = mmnorm,
                     newdata = OJ_dum)

#------------------------------------------

## Training and Testing

# Splitting the data into training and 
# testing sets using an 85/15 split rule

# Initialize random seed
set.seed(831) 

# Create list of training indices
sub <- createDataPartition(y = OJ_dum_mm$Purchase, # target variable
                           p = 0.85, # % in training
                           list = FALSE)

# Subset the transformed data
# to create the training (train)
# and testing (test) datasets
train <- OJ_dum_mm[sub, ] # create train dataframe
test <- OJ_dum_mm[-sub, ] # create test dataframe

#------------------------------------------

# In this lesson, we will use tensorflow and keras 
# for deep learning using neural networks.

# Tensorflow is an open-source library for machine 
# learning in python. Keras is a neural network API 
# written to run on top of Tensorflow. 

## Decisions that impact the model (hyperparameters) 
## include:
# - architecture (# hidden layers, # hidden nodes)
# - number of epochs
# - batch size
# - optimizer & learning rate
# - regularization & dropout (if needed)

## Analysis

# Defining a keras sequential model
# We use keras_model_sequential() to create
# a sequential model, where we add layers using %>%
model <- keras_model_sequential() %>% 
  layer_dense(units = 100, # Hidden Layer #1: 100 nodes
              input_shape = c(length(vars)),
              activation = "relu") %>%
  layer_dense(units = 64, # Hidden Layer #2: 64 nodes
              activation = "relu") %>%
  layer_dense(units = 20, # Hidden Layer #3: 20 nodes
              activation = "relu") %>%
  layer_dense(units = 10, # Hidden Layer #4: 10 nodes
              activation = "relu") %>%
  layer_dense(units = 1, # Output Layer: 1 node (binary classification)
              activation = "sigmoid")

# compile() is used to specify the optimizer,
# loss function to use and metrics to 
# evaluate during training. 

# The optimization approach is typically
# stochastic gradient descent, but the exact
# optimization algorithm used can be chosen.

model %>% compile(
  optimizer = 'adam', # adam is a popular optimizer (https://arxiv.org/pdf/1412.6980v8.pdf)
  loss = 'binary_crossentropy', # binary classification
  metrics = 'accuracy')

# We can view the ANN architecture using the
# summary() function on our model
summary(model)


## Model Training
history <- model %>% fit(
  x = as.matrix(train[ ,vars]), # input data must be matrix, use all variables not target
  y = train[ ,1], # target variable
  epochs = 20, # number of iterations over training set
  batch_size = 32, # batch size for updating
  validation_split = 0.1, # last 10% of observations are used as validation set
  verbose = 0 # sets the amount of console output
)

plot(history)


## Test Performance
model %>% evaluate(x = as.matrix(test[,vars]),
                   y = test[, 1])

# We can use the predict_classes() function to
# generate test predictions
preds_test <- as.numeric(model %>% predict(as.matrix(test[,vars])) %>% `>`(0.5))

# We can use the confusionMatrix() function to
# obtain performance
te_conf <- confusionMatrix(data = factor(preds_test), 
                           reference = factor(test[ ,1]),
                           positive = "1",
                           mode="everything")
te_conf


## Hyperparameter Tuning
# We can use the tfruns package to call a 
# separate R script to tune our hyperparameters
# The script is called "DL_Script.R" and the
# prepared data that the script will call is
# "DL_Tuning.RData". 

# The R script file is set up to allow the 
# following hyperparameters to be tuned:
# lrs: learning rate, default = 0.001
# batch: batch size, default = 32
# epoch_nums: number of epochs, default = 20
# nodes_1: number of nodes in Hidden Layer 1, default = 100
# nodes_2: number of nodes in Hidden Layer 2, default = 64
# nodes_3: number of nodes in Hidden Layer 3, default = 20
# nodes_3: number of nodes in Hidden Layer 4, default = 10

# First, we can train the default model and
# explore the results using the training_run()
# function from the tfruns package
training_run(file = "DL/DL_Script.R")

# Next, we can tune the model using a grid
# search using the tuning_run() function. 
# We send the results to a subfolder
# in our working directory named tuning.
# We want to maximize validation accuracy.
# We will demonstrate a random 5% of 
# combinations in the full grid search 
# defined below
hp_runs <- tuning_run("DL/DL_Script.R",
                      runs_dir = "tuning",
                      sample = 0.02, # try this proportion of combinations
                      flags = list(
                        batch = c(16, 32), # batch size
                        lrs = c(.01, .001), # learning rate
                        epoch_nums = c(10, 15, 20), # number of epochs
                        nodes_1 = c(100, 64), # number of nodes H1
                        nodes_2 = c(64, 32), # number of nodes H2
                        nodes_3 = c(20, 32), # number of nodes H3
                        nodes_4 = c(10, 20) # number of nodes H3
                      ))

# We can view the output as a spreadsheet
# using the View() function
View(hp_runs)

# After the optimal hyperparameters are found,
# we can train the chosen model multiple 
# times to obtain summary training performance
# before applying the model to our testing data.

# To remove logged run results from your
# local machine, use clean_runs() or
# purge_runs()

#------------------------------------------

