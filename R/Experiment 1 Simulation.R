library(compiler)
library(EnvStats)
enableJIT(3)

# set.seed(3)

log_dbeta_primitive <- function(x, shape1, shape2) {
  log_numerator <- lgamma(shape1 + shape2)
  log_denominator <- lgamma(shape1) + lgamma(shape2)
  
  log_result <- log_numerator - log_denominator + (shape1 - 1) * log(x) + (shape2 - 1) * log(1 - x)
  
  return(log_result)
}

save_simulation_data <- function(xx, simulation_result) {
  # Create a filename based on the input value xx
  filename <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"),"_simulation_", xx, ".Rdata")
  
  # Save the simulation_result to the file
  save(simulation_result, file = filename)
  
  # Print a message indicating the successful saving of the file
  cat("Simulation data saved to:", filename, "\n")
}

for (tt in 21:50) {
  RISK_STORE <- matrix(0,8,8)
  
  nn_choice <- 2^(9:15)
  gg_choice <- 1:8
  
  for (nn_sel in 1:7) {
    
    nn <- nn_choice[nn_sel]  
    
    for (gg_sel in 1:8)  {
      
      ## Sample from Triangle
      LL <- sample(1:2,nn,replace = TRUE)
      XX <- (LL==1)*runif(nn,0,0.4) + (LL==2)*runif(nn,0.6,1)
      YY <- rbeta(nn,0.5,0.5) 
      
      gg <- gg_choice[gg_sel]
      
      ## Sample from well-specified model
      # LL <- sample(1:gg,nn,replace = TRUE)
      # XX <- rep(0,nn)
      # alpha_true <- (1:gg)
      # beta_true <- (gg:1)
      # for (zz in 1:gg) {
      #   XX <- XX + (LL==zz)*rbeta(nn,alpha_true[zz],beta_true[zz])
      # }
      
      if (gg == 1) {
        # al_vec <- rexp(gg,1/gg)
        # be_vec <- rexp(gg,1/gg)
        # pi_vec <- rep(1/gg,gg)
        al_vec <- 1
        be_vec <- 1
        pi_vec <- 1
      }
      all_means <- seq(0.2,0.8,length.out=16)
      all_means <- all_means[order(abs(all_means-0.5),decreasing = TRUE)]
      if (gg > 1) {
        means <- all_means[1:gg]
        al_vec <- means*(means*(1-means)/0.01-1)
        be_vec <- (1-means)*(means*(1-means)/0.01-1)
        pi_vec <- c(rep(0.999/(gg-1),gg-1),0.001)
        # al_vec <- c(2,rexp(gg-1,1))
        # be_vec <- c(2,rexp(gg-1,1))
        # pi_vec <- c(1/2,rep(1/(gg-1),gg-1)/2)
        # al_vec <- c(al_vec,rexp(3))
        # be_vec <- c(be_vec,rexp(3))
        # pi_vec <- c(pi_vec*1/2,1/2)
      }
      
      
      iters <- 200
      tauX_mat <- matrix(0, nn, gg)
      tauY_mat <- matrix(0, nn, gg)
      tau_denom_X <- rep(0, nn)
      tau_denom_Y <- rep(0, nn)
      
      for (ss in 1:iters) {
        
        for (zz in 1:gg) {
          tauX_mat[, zz] <- pi_vec[zz] * exp(log_dbeta_primitive(XX, al_vec[zz], be_vec[zz]))
          
          tauY_mat[, zz] <- pi_vec[zz] * exp(log_dbeta_primitive(YY, al_vec[zz], be_vec[zz]))
        }
        
        tau_denom_X <- rowSums(tauX_mat) + dbeta(XX,0.5,0.5) #dunif(XX)
        tau_denom_Y <- rowSums(tauY_mat) + dbeta(YY,0.5,0.5) #dunif(YY)
        
        for (zz in 1:gg) {
          tauX_mat[,zz] <- tauX_mat[, zz]/tau_denom_X
          tauY_mat[,zz] <- tauY_mat[, zz]/tau_denom_Y
        }
        
        pi_vec <- colSums(tauX_mat + tauY_mat) / sum(tauX_mat + tauY_mat)
        
        update_params <- function(index) {
          # Precompute values outside the function
          tauX_values <- tauX_mat[, index]
          tauY_values <- tauY_mat[, index]
          
          func <- function(para) {
            exp_para1 <- exp(para[1])
            exp_para2 <- exp(para[2])
            # Vectorize the calculations
            result <- -sum(tauX_values * log_dbeta_primitive(XX, exp_para1, exp_para2) +
                             tauY_values * log_dbeta_primitive(YY, exp_para1, exp_para2)
            )
            
            return(result)
          }
          optim_result <- optim(log(c(al_vec[index], be_vec[index])), func,
                                lower = rep(exp(-5),2), upper = rep(exp(5),2),
                                method='L-BFGS-B',
                                control = list(maxit = 1000))
          c(exp(optim_result$par[1]), exp(optim_result$par[2]))
          
          # optim_result <- nlm(func,c(al_vec[index], be_vec[index]),steptol=1e-4,gradtol=1e-4)
          # c(exp(optim_result$estimate[1]), exp(optim_result$estimate[2]))
          
        }
        
        params <- t(sapply(1:gg, update_params))
        al_vec <- params[, 1]
        be_vec <- params[, 2]
        
        # print(c(pi_vec, al_vec, be_vec))
      }
      
      reso <- 100
      fitted_dens <- function(AL,BE,PI) {
        xx <- seq(0,1,length.out=reso)
        fitted <- rep(0,reso)
        for (ii in 1:reso) {
          for (zz in 1:length(AL)) {
            fitted[ii] <- fitted[ii] + PI[zz]*dbeta(xx[ii],AL[zz],BE[zz])
          }
        }
        return(fitted)
      }
      
      # fitted_loss <- function(xx,AL,BE,PI) {
      #   fitted <- 0
      #   for (zz in 1:length(AL)) {
      #     fitted <- fitted + PI[zz]*dbeta(xx,AL[zz],BE[zz])
      #   }
      #   return(fitted+1)
      # }
      
      fitted_loss <- function(xx, AL, BE, PI) {
        fitted <- sum(PI * dbeta(xx, AL, BE))
        return(fitted + dbeta(xx,0.5,0.5))
      }
      integrand_X <- function(xx) {
        log(fitted_loss(xx,al_vec,be_vec,pi_vec))*(dunif(xx,0,0.4)+dunif(xx,0.6,1))/2
      }
      vec_int_X <- Vectorize(integrand_X)
      integrand_Y <- function(xx) {
        log(fitted_loss(xx,al_vec,be_vec,pi_vec))*dbeta(xx,0.5,0.5)
      }
      vec_int_Y <- Vectorize(integrand_Y)
      
      ## Plot data
      hist(XX,probability = TRUE,'Scott')
      lines(seq(0,1,length.out=reso),
            fitted_dens(al_vec,be_vec,pi_vec))
      lines(density(XX,kernel = 'triangular'),col='red')
      
      # ## Compute risk
      # risk <- 0
      # for (ii in 1:NN_big) {
      #   risk <- risk - (1/NN_big)*log(fitted_loss(XX0[ii],al_vec,be_vec,pi_vec)) -
      #     (1/NN_big)*log(fitted_loss(YY0[ii],al_vec,be_vec,pi_vec))
      # }
      # RISK_STORE[nn_sel,gg_sel] <- risk
      
      # # Precompute constant values
      # NN_big_inv <- 1 / NN_big
      # 
      # # Create a vector of losses for XX0 and YY0
      # loss_XX <- sapply(XX0, fitted_loss, AL = al_vec, BE = be_vec, PI = pi_vec)
      # loss_YY <- sapply(YY0, fitted_loss, AL = al_vec, BE = be_vec, PI = pi_vec)
      # 
      # # Compute the risk using vectorized operations
      # risk <- -NN_big_inv * sum(log(loss_XX)) - NN_big_inv * sum(log(loss_YY))
      
      risk <- -integrate(vec_int_X,0,1,subdivisions = 1000)$value-integrate(vec_int_Y,0,1,subdivisions = 1000)$value
      
      # Store the result in RISK_STORE
      RISK_STORE[nn_sel, gg_sel] <- risk
      
      print(RISK_STORE)
    }
  }
  
  save_simulation_data(tt,RISK_STORE)
}



