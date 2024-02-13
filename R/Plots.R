# Define a function f1 that takes a parameter xx
f1 <- function(xx) {
  # Calculate the probability density function (PDF) for a uniform distribution
  # with values between 0 and 0.4, and between 0.6 and 1, and return their average.
  (dunif(xx, 0, 0.4) + dunif(xx, 0.6, 1)) / 2
}

# Define a true density function dtrue for a custom distribution
dtrue <- function(x) {
  if (x < 0) {
    output <- 0
  } 
  if (x > 1) {
    output <- 0
  }
  if (x >= 0 & x <= 1/2) {
    output <- 2 - 4 * x
  }
  if (x > 1/2 & x <= 1) {
    output <- -2 + 4 * x
  }
  return(output)
}

# Vectorize the dtrue function for use with vector inputs
dtrue_vec <- Vectorize(dtrue)

# Assign the vectorized dtrue function to f2
f2 <- dtrue_vec

# Plot f2 and f1 on the same graph
curve(f2, from = 0, to = 1, n = 1001, lwd = 2, lty = 2) # Plot f2
curve(f1, from = 0, to = 1, add = TRUE, n = 1001, lwd = 2) # Add f1 to the plot

# Plot the first function f2 with labels and fill
curve(f2, from = 0, to = 1, n = 1001, lwd = 2, lty = 2, ylab = 'density')

# Prepare to fill under f2 with a polygon
x_vals <- seq(0, 1, length.out = 1001)
y_vals <- f2(x_vals)
polygon(c(0, x_vals, 1), c(0, y_vals, 0), col = rgb(0, 0, 1, 0.3), border = NA)

# Add the second function f1 to the plot with fill
y_vals <- f1(x_vals)
polygon(c(0, x_vals, 1), c(0, y_vals, 0), col = rgb(1, 0, 0, 0.3), border = NA)

# Add a grid to the plot
grid()

# Assuming your data frame is named df
library(dplyr)

# Aggregate data in the data frame
agg_data <- FULL_FRAME %>%
  group_by(N, K) %>%
  summarize(AvgValue = mean(Value, na.rm = TRUE))

library(ggplot2)

# Modify the N column in agg_data to be a factor
agg_data$N <- as.factor(agg_data$N)

# Create a heatmap with cell borders using ggplot2
ggplot(agg_data, aes(x = N, y = K, fill = AvgValue)) +
  geom_tile(color = "black") + # Add cell borders
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "E1", x = "N", y = "K", fill = "Average") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
