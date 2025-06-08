# File: MovieLens_script.R
# Description: Script to train models, generate predictions, and calculate RMSE

# Load necessary packages
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")

library(tidyverse)
library(caret)
library(data.table)

# Set seed for reproducibility
set.seed(1, sample.kind = "Rounding")

# Download and prepare the data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies)
movies$movieId <- as.numeric(movies$movieId)

movielens <- left_join(ratings, movies, by = "movieId")

# Split edx and final hold-out test set
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
final_holdout_test <- movielens[test_index, ] %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Clean up
rm(dl, ratings, movies, test_index, movielens)

# Create a training and validation set from edx
set.seed(1, sample.kind = "Rounding")
index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-index, ]
temp <- edx[index, ]

# Ensure userId and movieId in validation are also in training set
validation <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
train_set <- train_set %>%
  semi_join(validation, by = "movieId") %>%
  semi_join(validation, by = "userId")

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Naive Model
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)
rmse_results <- tibble(method = "Naive Mean Model", RMSE = naive_rmse)

# Movie Effect Model
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu_hat + b_i) %>%
  pull(pred)
movie_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie Effect Model", RMSE = movie_rmse))

# Movie + User Effects Model
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)
user_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User Effects Model", RMSE = user_rmse))

# Display RMSE results
print(rmse_results)

# At this point, you would continue with regularization and/or matrix factorization
# and then apply the final model to final_holdout_test and report the RMSE

