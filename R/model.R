#' Model Function
#'
#' This function builds the deep neural network metamodel architecture using torch.
#' @param dataset Training and test data
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param ext.dir External directory (full path)
#' @return A deep neural network metamodel of Monte Carlo radiation transport code simulation data
#' @export
#' @import torch
#' @import magrittr
Model <- function(
    dataset,
    layers = '8192-256-256-256-256-16',
    loss = 'sse',
    opt.alg = 'adam',
    learning.rate = 0.00075,
    ext.dir) {
  
  # Parse layers string into integer vector
  layer_sizes <- strsplit(layers, '-') %>% 
    unlist() %>% 
    as.integer()
  
  # Define the neural network structure
  DNN <- nn_module(
    initialize = function() {
      # Get input dimension from dataset
      self$input_dim <- dim(dataset$training.df)[2]
      
      # Create sequential container for layers
      self$network <- nn_sequential()
      
      # Add input layer
      self$network$add_module(
        "input",
        nn_linear(self$input_dim, layer_sizes[1])
      )
      self$network$add_module(
        "relu1",
        nn_relu()
      )
      
      # Add hidden layers
      for (i in seq_len(length(layer_sizes) - 1)) {
        self$network$add_module(
          sprintf("linear%d", i + 1),
          nn_linear(layer_sizes[i], layer_sizes[i + 1])
        )
        self$network$add_module(
          sprintf("relu%d", i + 1),
          nn_relu()
        )
      }
      
      # Add output layer (linear activation by default)
      self$network$add_module(
        "output",
        nn_linear(layer_sizes[length(layer_sizes)], 1)
      )
    },
    
    forward = function(x) {
      self$network(x)
    }
  )
  
  # Create model instance
  model <- DNN()
  
  # Define loss function
  criterion <- if (loss == 'sse') {
    function(pred, target) {
      torch_sum((pred - target)^2)
    }
  } else {
    stop("Unsupported loss function")
  }
  
  # Define available optimizers
  optimizers <- list(
    adadelta = function() optim_adadelta(model$parameters, lr = learning.rate),
    adagrad = function() optim_adagrad(model$parameters, lr = learning.rate),
    adam = function() optim_adam(model$parameters, lr = learning.rate),
    rmsprop = function() optim_rmsprop(model$parameters, lr = learning.rate)
  )
  
  # Validate and select optimizer
  if (!opt.alg %in% names(optimizers)) {
    stop(sprintf(
      "Invalid optimizer '%s'. Available optimizers: %s",
      opt.alg,
      paste(names(optimizers), collapse = ", ")
    ))
  }
  
  # Create optimizer
  optimizer <- optimizers[[opt.alg]]()
  
  # Return model components
  list(
    model = model,
    criterion = criterion,
    optimizer = optimizer
  )
}

#' Training function for the model
#' @param model_obj Model object returned by Model()
#' @param dataset Training dataset
#' @param epochs Number of training epochs
#' @param batch_size Batch size for training
#' @param verbose Whether to print progress
train_model <- function(model_obj, dataset, epochs = 100, batch_size = 32, verbose = TRUE) {
  # Extract components
  model <- model_obj$model
  criterion <- model_obj$criterion
  optimizer <- model_obj$optimizer
  
  # Convert data to torch tensors
  x_train <- torch_tensor(as.matrix(dataset$training.df))
  y_train <- torch_tensor(as.matrix(dataset$training.labels))
  
  # Set model to training mode
  model$train()
  
  # Training loop
  n_samples <- dim(x_train)[1]
  
  for (epoch in 1:epochs) {
    # Mini-batch training
    total_loss <- 0
    
    for (b in seq(1, n_samples, batch_size)) {
      # Get batch indices
      end_idx <- min(b + batch_size - 1, n_samples)
      batch_x <- x_train[b:end_idx, ]
      batch_y <- y_train[b:end_idx, ]
      
      # Forward pass
      optimizer$zero_grad()
      output <- model(batch_x)
      loss <- criterion(output, batch_y)
      
      # Backward pass and optimize
      loss$backward()
      optimizer$step()
      
      total_loss <- total_loss + loss$item()
    }
    
    # Print progress
    if (verbose && epoch %% 10 == 0) {
      avg_loss <- total_loss / ceiling(n_samples / batch_size)
      cat(sprintf("Epoch %d/%d, Average Loss: %.4f\n", 
                 epoch, epochs, avg_loss))
    }
  }
}
