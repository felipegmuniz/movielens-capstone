# movieLens_script.R
# Author: Felipe Muniz
# Description: MovieLens 10M Capstone assignment - model training, predictions, and RMSE calculation

# Load required packages
if (!require(tidyverse)) install.packages("tidyverse", dependencies = TRUE)
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(data.table)) install.packages("data.table", dependencies = TRUE)

library(tidyverse)
library(caret)
library(data.table)

# Set seed for reproducibility
set.seed(1, sample.kind = "Rounding")

# Download and prepare MovieLens 10M dataset
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies)
movies$movieId <- as.numeric(movies$movieId)

movielens <- left_join(ratings, movies, by = "movieId")

# Create edx and final holdout test sets
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
final_holdout_test <- movielens[test_index, ] %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

rm(dl, ratings, movies, test_index, movielens)

# Create training and validation sets from edx
set.seed(1, sample.kind = "Rounding")
train_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-train_index, ]
temp <- edx[train_index, ]

validation <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

train_set <- train_set %>%
  semi_join(validation, by = "movieId") %>%
  semi_join(validation, by = "userId")

# Define RMSE function (quick helper)
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Baseline/Naive Model
global_avg <- mean(train_set$rating)  # overall average
naive_preds <- rep(global_avg, nrow(validation))
naive_rmse <- rmse(validation$rating, naive_preds)

results <- data.frame(
  Method = "Naive Average",
  RMSE = naive_rmse
)

# Movie Effect Model
# Estimates how each movie deviates from the global average
movie_bias <- train_set %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - global_avg))

validation_with_movie <- validation %>%
  left_join(movie_bias, by = "movieId")

validation_with_movie <- validation_with_movie %>%
  mutate(pred = global_avg + movie_effect)

movie_rmse <- rmse(validation_with_movie$rating, validation_with_movie$pred)

results <- rbind(results, data.frame(
  Method = "Movie Effect Model",
  RMSE = movie_rmse
))

# Movie + User Effects Model 
# Estimates how each user deviates after accounting for movie effect
user_bias <- train_set %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - global_avg - movie_effect))

validation_with_all <- validation_with_movie %>%
  left_join(user_bias, by = "userId") %>%
  mutate(pred = global_avg + movie_effect + user_effect)

user_rmse <- rmse(validation_with_all$rating, validation_with_all$pred)

results <- rbind(results, data.frame(
  Method = "Movie + User Effect Model",
  RMSE = user_rmse
))

# Show RMSE results
print(results)

# Final Evaluation on Hold-Out Test Set
# Recalculate biases using full edx dataset
global_avg_edx <- mean(edx$rating)

movie_bias_edx <- edx %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - global_avg_edx))

user_bias_edx <- edx %>%
  left_join(movie_bias_edx, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - global_avg_edx - movie_effect))

# Generate predictions
final_predictions <- final_holdout_test %>%
  left_join(movie_bias_edx, by = "movieId") %>%
  left_join(user_bias_edx, by = "userId") %>%
  mutate(pred = global_avg_edx + movie_effect + user_effect) %>%
  pull(pred)

# Replace NAs (for unseen movieId or userId) with global average
final_predictions[is.na(final_predictions)] <- global_avg_edx

final_rmse <- rmse(final_holdout_test$rating, final_predictions)

# Show result
cat(sprintf("\nFinal RMSE on Hold-Out Test Set: %.5f\n", final_rmse))

