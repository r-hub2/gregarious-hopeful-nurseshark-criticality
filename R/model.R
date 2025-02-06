#' Build deep neural network metamodel architecture using torch
#' @param dataset Training and test data
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learn.rate Learning rate
#' @param ext.dir External directory (full path)
#' @return A deep neural network metamodel
#' @export
#' @import torch
#' @import magrittr

Model <- function(dataset,
                 layers = '8192-256-256-256-256-16',
                 loss = 'sse',
                 opt.alg = 'adam',
                 learn.rate = 0.00075,
                 ext.dir) {
  
  # Parse layers string
  layer_sizes <- strsplit(layers, '-') %>% unlist() %>% as.integer()
  
  # Build sum of squared errors (SSE) loss function
  if (loss == 'sse') {
    loss <- function(y_true, y_pred) torch::sum((y_true - y_pred)^2)
  }
  
  # Define model architecture using nn_module
  model <- nn_module(
    initialize = function() {
      # Input dimension from dataset
      input_dim <- dim(dataset$training.df)[2]
      
      # Create a list to store all layers
      self$layers <- nn_module_list()
      
      # Add input layer
      self$layers$append(nn_linear(input_dim, layer_sizes[1]))
      
      # Add hidden layers
      for (i in 1:(length(layer_sizes) - 1)) {
        self$layers$append(nn_linear(layer_sizes[i], layer_sizes[i + 1]))
      }
      
      # Add output layer
      self$output <- nn_linear(layer_sizes[length(layer_sizes)], 1)
    },
    
    forward = function(x) {
      # Apply ReLU activation to all hidden layers
      for (layer in self$layers) {
        x <- torch_relu(layer(x))
      }
      # Final output layer without activation
      x <- self$output(x)
      return(x)
    }
  )
  
  # Define optimizer based on selected algorithm
  optimizer <- switch(opt.alg,
                     "adadelta" = optim_adadelta(model$parameters, lr = learn.rate),
                     "adagrad" = optim_adagrad(model$parameters, lr = learn.rate),
                     "adam" = optim_adam(model$parameters, lr = learn.rate),
                     "rmsprop" = optim_rmsprop(model$parameters, lr = learn.rate),
                     stop("Unsupported optimizer algorithm"))
  
  return(list(model = model, criterion = loss, optimizer = optimizer))

}