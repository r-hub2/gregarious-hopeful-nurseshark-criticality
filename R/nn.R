#' NN Function
#'
#' This function trains an ensemble of deep neural networks to predict keff values using torch.
#' @param batch.size Batch size
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dataset Training and test data
#' @param ensemble.size Number of deep neural networks in the ensemble
#' @param epochs Number of training epochs
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param val.split Validation split
#' @param overwrite Boolean (TRUE/FALSE) that determines if files should be overwritten
#' @param replot Boolean (TRUE/FALSE) that determines if .png files should be replotted
#' @param reweight Boolean (TRUE/FALSE) that determines if metamodel weights should be recalculated
#' @param verbose Boolean (TRUE/FALSE) that determines if training output should be displayed
#' @param ext.dir External directory (full path)
#' @param training.dir Training directory (full path)
#' @return A list of lists containing an ensemble of deep neural networks and weights
#' @export
#' @import torch
#' @import magrittr
NN <- function(
    batch.size = 8192,
    code = 'mcnp',
    dataset,
    ensemble.size = 5,
    epochs = 1500,
    layers = '8192-256-256-256-256-16',
    loss = 'sse',
    opt.alg = 'adam',
    learning.rate = 0.00075,
    val.split = 0.2,
    overwrite = FALSE,
    replot = TRUE,
    reweight = FALSE,
    verbose = FALSE,
    ext.dir,
    training.dir = NULL) {
  
  # Initialize dataset if not provided
  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)
  
  # Set up directories
  if (is.null(training.dir)) training.dir <- paste0(ext.dir, '/training')
  model.dir <- paste0(training.dir, '/model')
  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create settings dataframe
  new.settings <- data.frame(V1 = c(
    'model settings',
    paste0('batch size: ', batch.size),
    paste0('code: ', code),
    paste0('ensemble size: ', ensemble.size),
    paste0('epochs: ', epochs),
    paste0('layers: ', layers),
    paste0('loss: ', loss),
    paste0('optimization algorithm: ', opt.alg),
    paste0('learning rate: ', learning.rate),
    paste0('validation split: ', val.split),
    paste0('external directory: ', ext.dir),
    paste0('training directory: ', training.dir)))
  
  # Check metamodel settings
  if (file.exists(paste0(training.dir, '/model-settings.txt'))) {
    old.settings <- utils::read.table(paste0(training.dir, '/model-settings.txt'), sep = '\n') %>% as.data.frame()
    if (!identical(new.settings[-4, ], old.settings[-4, ])) {
      if (overwrite) {
        unlink(model.dir, recursive = TRUE)
        unlink(remodel.dir, recursive = TRUE)
        utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), 
                          quote = FALSE, row.names = FALSE, col.names = FALSE)
        dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)
        dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)
      } else {
        stop('files could not be overwritten', call. = FALSE)
      }
    } else {
      utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), 
                        quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
  } else {
    utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), 
                      quote = FALSE, row.names = FALSE, col.names = FALSE)
    old.settings <- new.settings
  }
  
  # Define SSE loss function
  if (loss == 'sse') {
    criterion <- function(pred, target) torch_sum((pred - target)^2)
  }
  
  # Helper function to save model
  save_model <- function(model, path) {
    torch_save(model$state_dict(), path)
  }
  
  # Helper function to load model
  load_model <- function(model, path) {
    model$load_state_dict(torch_load(path))
    model
  }
  
  # Training function with validation
  train_model <- function(model, dataset, batch_size, epochs, val_split, verbose, save_path = NULL) {
    # Convert data to tensors
    x_train <- torch_tensor(as.matrix(dataset$training.df))
    y_train <- torch_tensor(as.matrix(dataset$training.data$keff))
    
    # Split into train and validation
    n_samples <- dim(x_train)[1]
    n_val <- floor(n_samples * val_split)
    indices <- torch_randperm(n_samples)
    
    train_indices <- indices[1:(n_samples - n_val)]
    val_indices <- indices[(n_samples - n_val + 1):n_samples]
    
    x_train_split <- x_train[train_indices, ]
    y_train_split <- y_train[train_indices]
    x_val <- x_train[val_indices, ]
    y_val <- y_train[val_indices]
    
    # Training loop
    history <- list(
      loss = numeric(epochs),
      val_loss = numeric(epochs),
      mae = numeric(epochs),
      val_mae = numeric(epochs)
    )
    
    for (epoch in 1:epochs) {
      model$train()
      total_loss <- 0
      total_mae <- 0
      
      # Mini-batch training
      for (b in seq(1, dim(x_train_split)[1], batch_size)) {
        end_idx <- min(b + batch_size - 1, dim(x_train_split)[1])
        batch_x <- x_train_split[b:end_idx, ]
        batch_y <- y_train_split[b:end_idx]
        
        optimizer$zero_grad()
        output <- model(batch_x)
        loss <- criterion(output, batch_y)
        loss$backward()
        optimizer$step()
        
        total_loss <- total_loss + loss$item()
        total_mae <- total_mae + torch_mean(torch_abs(output - batch_y))$item()
      }
      
      # Validation
      model$eval()
      with_no_grad({
        val_output <- model(x_val)
        val_loss <- criterion(val_output, y_val)$item()
        val_mae <- torch_mean(torch_abs(val_output - y_val))$item()
      })
      
      # Save history
      history$loss[epoch] <- total_loss / ceiling(dim(x_train_split)[1] / batch_size)
      history$val_loss[epoch] <- val_loss
      history$mae[epoch] <- total_mae / ceiling(dim(x_train_split)[1] / batch_size)
      history$val_mae[epoch] <- val_mae
      
      # Save checkpoint if path provided
      if (!is.null(save_path)) {
        save_model(model, paste0(save_path, '/model_', epoch, '.pt'))
      }
      
      # Print progress
      if (verbose && epoch %% 10 == 0) {
        cat(sprintf("Epoch %d/%d - loss: %.4f - val_loss: %.4f - mae: %.4f - val_mae: %.4f\n",
                   epoch, epochs, history$loss[epoch], history$val_loss[epoch],
                   history$mae[epoch], history$val_mae[epoch]))
      }
    }
    
    history
  }
  
  # Load or train metamodel
  can_load_existing <- file.exists(paste0(training.dir, '/metamodel.RData')) &&
    identical(new.settings[-4, ], old.settings[-4, ]) &&
    ensemble.size == dim(utils::read.csv(paste0(training.dir, '/test-mae.csv')))[1] &&
    ensemble.size <= length(list.files(path = model.dir, pattern = "*.pt")) &&
    !reweight
  
  if (can_load_existing) {
    # Load existing metamodel
    wt <- min.wt <- numeric()
    load(paste0(training.dir, '/metamodel.RData'))
    
    metamodel <- vector("list", ensemble.size)
    for (i in 1:ensemble.size) {
      model <- Model(dataset, layers, loss, opt.alg, learning.rate, ext.dir)
      metamodel[[i]] <- load_model(model, paste0(remodel.dir, '/', i, '-', min.wt[[1]][[i]], '.pt'))
      wt[i] <- min.wt[[2]][[i]]
    }
  } else {
    # Train new metamodel
    metamodel <- vector("list", ensemble.size)
    history <- vector("list", ensemble.size)
    
    # Initial training
    for (i in 1:ensemble.size) {
      metamodel[[i]] <- Model(dataset, layers, loss, opt.alg, learning.rate, ext.dir)
      history[[i]] <- train_model(metamodel[[i]], dataset, batch_size, epochs, val.split, verbose)
      Plot(i = i, history = history[[i]], plot.dir = model.dir)
      save_model(metamodel[[i]], paste0(model.dir, '/', i, '.pt'))
    }
    
    # Retraining with shorter epochs
    for (i in 1:ensemble.size) {
      history[[i]] <- train_model(metamodel[[i]], dataset, batch_size, epochs %/% 10, 
                                 val.split, verbose, paste0(remodel.dir, '/', i))
      Plot(i = i, history = history[[i]], plot.dir = remodel.dir)
    }
    
    # Calculate weights
    wt <- Test(dataset, ensemble.size, loss, ext.dir, training.dir)
  }
  
  list(metamodel, wt)
}
