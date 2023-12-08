library(tidyverse)
library(tidymodels)
library(vroom)
library(embed) # for target encoding
library(ranger)
library(rpart)
library(discrim)
library(naivebayes)
library(kknn)
library(doParallel)
library(themis)
library(stacks)
library(kernlab)
library(keras)
library(tensorflow)
library(bonsai)
library(lightgbm)
library(dbarts)

train <- vroom("./training.csv", na=c("","NULL", "NA")) %>%
  mutate(IsBadBuy=factor(IsBadBuy))

test <- vroom("./test.csv", na=c("", "NA", "NULL"))

# Count the number of NA values in each column
na_count <- train %>%
  summarise_all(~sum(is.na(.)))

# View the result
print(na_count)
dim(train)

# Recipe ------------------------------------------------------------------
my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>%
  step_unknown(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(balance_recipe)
baked <- bake(prepped_recipe, new_data = train)

na_count_baked <- baked %>%
  summarise_all(~sum(is.na(.)))

# Should I use balancing?
train %>% 
  count(IsBadBuy)

# Penalized Logistic Regression
pen_log_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine('glmnet')

pen_log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_log_mod)

pen_tuning_grid <- grid_regular(penalty(),
                                mixture(),
                                levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats = 1)

cv_results <- tune_grid(pen_log_wf,
                        resamples = folds,
                        grid = pen_tuning_grid,
                        metrics = metric_set(roc_auc))

collect_metrics(cv_results)

best_tune <- select_best(cv_results, 'roc_auc')

final_wf <- pen_log_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

pen_log_preds <- predict(final_wf, new_data = test, type = 'prob')
pen_log_preds

kicked_predictions <- final_wf %>% 
  predict(new_data = test, type = 'prob') %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>% 
  select(RefId, IsBadBuy)

kicked_predictions

vroom_write(kicked_predictions, file = './Penalized_Log_Regression.csv', delim = ",")

# Random Forest -----------------------------------------------------------
rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees=500) %>% # or 1000
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                        min_n(),
                                        levels = 5) ## L^2 total tuning possibilities

## Split data for CV
forest_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- rand_forest_workflow %>%
  tune_grid(resamples = forest_folds,
            grid = rand_forest_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
forest_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_forest_wf <- rand_forest_workflow %>%
  finalize_workflow(forest_bestTune) %>%
  fit(data = train)

rand_forest_predictions <- final_forest_wf %>% 
  predict(new_data = test, type = 'prob') %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>% 
  select(RefId, IsBadBuy)

rand_forest_predictions

vroom_write(rand_forest_predictions, file = './Random_Forest.csv', delim = ",")


# KNN ---------------------------------------------------------------------
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# cross validation
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)

knn_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = knn_folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

knn_bestTune <- CV_results %>%
  select_best("roc_auc")

# finalize workflow
final_knn_wf <- knn_wf %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data = train)

knn_predictions <- final_knn_wf %>% 
  predict(new_data = test, type = 'prob') %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>% 
  select(RefId, IsBadBuy)

knn_predictions

vroom_write(knn_predictions, file = './knn.csv', delim = ",")

# Naive Bayes -------------------------------------------------------------
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

## Split data for CV
nb_folds <- vfold_cv(train, v = 5, repeats = 1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
nb_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_nb_wf <- nb_wf %>%
  finalize_workflow(nb_bestTune) %>%
  fit(data = train)

nb_predictions <- final_nb_wf %>% 
  predict(new_data = test, type = 'prob') %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>% 
  select(RefId, IsBadBuy)

nb_predictions

vroom_write(nb_predictions, file = './nb.csv', delim = ",")


# Stack Model -------------------------------------------------------------

folds <- vfold_cv(train, v = 5, repeats=2)
untunedModel <- control_stack_grid()

nb_models <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

pen_log_models <- pen_log_wf %>%
  tune_grid(resamples=folds,
            grid=pen_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

# Specify with models to include
my_stack <- stacks() %>%
  add_candidates(pen_log_models) %>%
  add_candidates(nb_models)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

predictions <- stack_mod %>%
  predict(new_data = test,
          type = "prob")

submission

vroom_write(submission, file = './stack_nb_pen_log.csv', delim = ",")

# BART --------------------------------------------------------------------
# Create Model
bart_mod <- parsnip::bart(trees = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

# Create workflow
bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

# Tuning Grid
bart_tuning_grid <- grid_regular(trees(),
                                 levels = 5)

# Create folds
bart_folds <- vfold_cv(train, v = 5, repeats = 1)

# Cross Validation
CV_results <- bart_wf %>%
  tune_grid(resamples = bart_folds,
            grid = bart_tuning_grid,
            metrics = metric_set(accuracy))

# Find the best tune for parameters
bart_bestTune <- CV_results %>%
  select_best("accuracy")

# Finalize workflow
final_bart_wf <- bart_wf %>%
  finalize_workflow(bart_bestTune) %>%
  fit(data = train)

# Predict and format
kicked_predictions <- final_bart_wf %>% 
  predict(new_data = test, type = 'prob') %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>% 
  select(RefId, IsBadBuy)

# View predictions
kicked_predictions

#vroom_write(kicked_predictions, file = './submission.csv', delim = ",")


