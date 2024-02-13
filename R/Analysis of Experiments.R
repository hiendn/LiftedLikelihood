# Step 1: Set the directory to the folder containing the .Rdata files
# Replace 'path_to_folder' with the path to your folder
# setwd("path_to_folder")

# Step 2: Get a list of all .Rdata files in the folder
file_list <- list.files(pattern = "\\.Rdata$")

# Step 3: Initialize an empty list to store the loaded data
LIST <- list()

# Step 4: Loop through each file, load and save it in the LIST
for (i in seq_along(file_list)) {
  file_path <- file_list[i]
  data <- load(file_path)
  LIST[[i]] <- get(data)
}

# Now, the LIST contains the data from all the .Rdata files in the folder.
# You can access each element using LIST[[1]], LIST[[2]], etc.

nn_choice <- 2^(10:15)
gg_choice <- 1:8

simulation_result <- LIST[[1]]
# Example matrix X
X <- simulation_result[-c(1,8),-1]

# Get the row and column indices
rows <- row(X)
cols <- col(X)

# Create a data frame
df <- data.frame(N = as.vector(nn_choice[rows]), 
                 K = as.vector(gg_choice[cols]+1), 
                 Value = as.vector(X))

FULL_FRAME <- df

for (ll in 2:50) {
  simulation_result <- LIST[[ll]]
  # Example matrix X
  X <- simulation_result[-c(1,8),-1]
  
  # Get the row and column indices
  rows <- row(X)
  cols <- col(X)
  
  # Create a data frame
  df <- data.frame(N = as.vector(nn_choice[rows]), 
                   K = as.vector(gg_choice[cols]+1), 
                   Value = as.vector(X))
  
  FULL_FRAME <- rbind(FULL_FRAME,df)
}

CONTROL <- nls.control(maxiter = 10000, tol = 1e-05, minFactor = 1/1000000000,
            printEval = FALSE, warnOnly = TRUE, scaleOffset = 0,
            nDcentral = TRUE)
NLS <- nls(Value ~ a0 +a1/((K+2)^b1) + a2/(N^b2),data=FULL_FRAME,control = CONTROL,
           start=list(a0=-2,a1=2,a2=2,b1=1,b2=0.5))
library(sandwich)
library(lmtest)

coeftest(NLS,vcov. = sandwich)

coefci(NLS,vcov. = sandwich)

plot(residuals(NLS)~log(FULL_FRAME$N))
abline(h=0)
plot(residuals(NLS)~FULL_FRAME$K)
abline(h=0)

aggregate(FULL_FRAME$Value,list(FULL_FRAME$N),FUN=median)
