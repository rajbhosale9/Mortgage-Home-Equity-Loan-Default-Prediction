library(keras)


FLAGS <- flags(
  flag_integer(name = "batch", 
               default = 32, 
               description = 'batch size'),
  flag_numeric(name = "lrs", 
               default = .001, 
               description = "learning rate"),
  flag_integer(name = "epoch_nums",
               default = 20,
               description = "number of epochs"),
  flag_integer(name = "nodes_1",
               default = 100,
               description = "number of nodes in H1"),
  flag_integer(name = "nodes_2",
               default = 64,
               description = "number of nodes in H2"),
  flag_integer(name = "nodes_3",
               default = 20,
               description = "number of nodes in H3"),
  flag_integer(name = "nodes_4",
               default = 10,
               description = "number of nodes in H4"))

load("DL/DL_Tuning.RData")
train <- as.matrix(train)
test <- as.matrix(test)

## Deep Learning Analysis
#defining a keras sequential model
model <- keras_model_sequential() %>% 
  layer_dense(units = FLAGS$nodes_1, # Hidden Layer #1
              input_shape = c(length(vars)),
              activation = "relu") %>%
  layer_dense(units = FLAGS$nodes_2, # Hidden Layer #2
              activation = "relu") %>%
  layer_dense(units = FLAGS$nodes_3, # Hidden Layer #3
              activation = "relu") %>%
  layer_dense(units = FLAGS$nodes_4, # Hidden Layer #4
              activation = "relu") %>%
  layer_dense(units = 1, # Output Layer: 1 node (binary classification)
              activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$lrs),
  loss = 'binary_crossentropy',
  metrics = 'accuracy')


## Model Training
hist <- model %>% fit(
  x = train[ ,vars], # use all variables not target
  y = train[ ,1], # target variable
  epochs = FLAGS$epoch_nums, # number of iterations over training set
  batch_size = FLAGS$batch, # batch size for updating
  validation_split = 0.1, # 10% of observations are used as validation set
  verbose = 0 # sets the amount of console output
)


## Test Performance
# score <- model %>% evaluate(x = test[,vars],
#                   y = test[,1])
