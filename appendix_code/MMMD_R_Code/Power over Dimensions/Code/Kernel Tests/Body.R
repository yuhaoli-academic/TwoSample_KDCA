################################################################################
################# Driver Code for Power over Dimensions ########################
################################################################################

# Calling the functions required for experiment
source("Functions.R")

start <- Sys.time()
# Number of repetitions
n.rep <- 50
# Sample size
n <- 100
# Dimension vector of data
d <- c(5, 10, 25, 50,75,100,150)

# Change the following parameters for different comparisons.
################################################################################
# probability of mixture
p <- 1
# parameter for sigma0 matrix generation
sigma.param <- 0.5
# parameter for sigma1 = c*sigma0 matrix generation
sigma.mult <- 1.22
# parameter for mean value for H1 distribution
mu.param <- 0
################################################################################
# Choice of kernel for single and multiple kernel tests
# poissble choices: 
# single - "GAUSS" or "LAP" for gaussian or laplacian
# multiple - "MINMAX", "GEXP" or "MIXED" for gaussian kernel with min-max, 
# exponential bandwidth choice, or mixture of gaussian and laplace kernel and
# "LAP" for laplace kernel with exponential bandwidth choice.

# First two coordinate for single, Last three for multiple
kernel.choice <- c("LAP", "GAUSS", "LAP", "GEXP", "MIXED")

# Libraries for parallelising
library(foreach)
library(doParallel)
library(snow)

#cores <- strtoi(Sys.getenv("NSLOTS"))-1
#cl <- makeCluster(cores, methods = FALSE, type = "MPI")

#registerDoParallel(cl)

# Uncomment below lines and comment above lines if not using MPI

cores <- detectCores()
cl <- makeCluster(cores[1]-1) #not to overload your computer

registerDoParallel(cl)


# Running experiment
out.d <- c()
for (iter in 1:n.rep){
  # storing power values for particular iteration
  out.d.iter <- power.d(n, sigma.param = sigma.param, sigma.mult = sigma.mult,
                        mu.param = mu.param, d.seq = d, p = p,
                        kernel.choice = kernel.choice, n.iter = 500)
  out.d <- c(out.d, out.d.iter)
  print (iter)
}

# Making the output a matrix for ease of access
out.d <- as.matrix(as.data.frame(out.d))
end <- Sys.time()
end-start

stopCluster(cl)

# Indices of Single and Multiple Kernel outputs
single.power.d1 <- 2+6*(0:(n.rep-1))
single.power.d2 <- 3+6*(0:(n.rep-1))
multi.power.d1 <- 4+6*(0:(n.rep-1))
multi.power.d2 <- 5+6*(0:(n.rep-1))
multi.power.d3 <- 6*(1:n.rep)

# Matrices for storing the power values over varying dimensions
power.single.mat1 <- matrix(0, nrow = length(d), ncol = n.rep)
power.single.mat2 <- matrix(0, nrow = length(d), ncol = n.rep)
power.multi.mat1 <- matrix(0, nrow = length(d), ncol = n.rep)
power.multi.mat2 <- matrix(0, nrow = length(d), ncol = n.rep)
power.multi.mat3 <- matrix(0, nrow = length(d), ncol = n.rep)

for (k in 1:length(d)){
  power.single.mat1[k,] <- out.d[k,single.power.d1]
  power.single.mat2[k,] <- out.d[k,single.power.d2]
  power.multi.mat1[k,] <- out.d[k,multi.power.d1]
  power.multi.mat2[k,] <- out.d[k,multi.power.d2]
  power.multi.mat3[k,] <- out.d[k,multi.power.d3]
}

# Keeping the vector of dimensions as a reference in all the outputs
power.single.mat1 <- cbind(d,power.single.mat1)
power.single.mat2 <- cbind(d,power.single.mat2)
power.multi.mat1 <- cbind(d,power.multi.mat1)
power.multi.mat2 <- cbind(d,power.multi.mat2)
power.multi.mat3 <- cbind(d,power.multi.mat3)

# Storing the power values
write.csv(power.single.mat1, file = "SinglePower-LAP.csv")
write.csv(power.single.mat2, file = "SinglePower-GAUSS.csv")
write.csv(power.multi.mat1, file = "MultiPower-LAP.csv")
write.csv(power.multi.mat2, file = "MultiPower-GEXP.csv")
write.csv(power.multi.mat3, file = "MultiPower-MIXED.csv")
