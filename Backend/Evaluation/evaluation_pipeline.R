library(tidyverse)
library(caret)
library(GGally)
library(ggfortify)
library(scales)
library(RColorBrewer)

setwd("/Users/heidemann/documents/private/Project_Seminar/Backend")


#### 1) Import Predictions ####
df_forecasts <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_forecasts_30_01.csv")
df_metrics <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_metrics_30_01.csv")

# Rename First Column:
df_forecasts <- df_forecasts %>% 
  rename(
    day_num = names(df_forecasts)[1]
  ) %>% 
  mutate(
    day_num = day_num + 1
  )

#### 2) Residuals Analysis ####

## Compute relative residuals:
df_forecasts <- df_forecasts %>% mutate(
  rel_residuals_Diff_Eq_Last_Beta = residuals_Diff_Eq_Last_Beta / y_val,
  rel_residuals_Diff_Eq_Ml_Beta = residuals_Diff_Eq_ML_Beta / y_val,
  rel_residuals_Sarima = residuals_Sarima / y_val,
  rel_residuals_ensemble = residuals_Ensemble / y_val,
)

### Last_Beta
## Diff_Eq_Last_Beta: Visualize residuals on forecasting day 14:
df_forecasts %>% 
  filter(day_num == 14) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_Last_Beta), 
                 binwidth = 5) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-300,300))

## Diff_Eq_Last_Beta: Visualize residuals on forecasting day 1:
df_forecasts %>% 
  filter(day_num == 1) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_Last_Beta), 
                 binwidth = 1) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-60,60))

### ML-Beta
## Diff_ML_Last_Beta: Visualize residuals on forecasting day 14:
df_forecasts %>% 
  filter(day_num == 14) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_ML_Beta), 
                 binwidth = 5) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-300,300))

## Diff_ML_Last_Beta: Visualize residuals on forecasting day 1:
df_forecasts %>% 
  filter(day_num == 1) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_ML_Beta), 
                 binwidth = 1) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-60,60))


  
## Diff_Eq_Last_Beta: Visualize average absolute residuals per day:
df_forecasts %>% 
  group_by(day_num) %>% 
  summarise(
    perc_20_quantile_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.2),
    average_residual = mean(residuals_Diff_Eq_Last_Beta),
    perc_80_quantile_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.8),
  ) %>% 
  ggplot() +
  geom_line(aes(x=day_num, y=perc_20_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  geom_line(aes(x=day_num, y=average_residual), size=2, color='darkgreen') +
  geom_line(aes(x=day_num, y=perc_80_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  ggtitle("Residuals", subtitle='Differential Equation Last Beta') +
  labs(y = "Absolute Residual", x="Day", fill='Anruf Empfehlung') +
  theme(text = element_text(size = 12), legend.position = "right")


## Diff_Eq_Ml_Beta: Visualize average absolute residuals per day:
df_forecasts %>% 
  group_by(day_num) %>% 
  summarise(
    perc_20_quantile_residual = quantile(residuals_Diff_Eq_ML_Beta, 0.2),
    average_residual = mean(residuals_Diff_Eq_Last_Beta),
    perc_80_quantile_residual = quantile(residuals_Diff_Eq_ML_Beta, 0.8),
  ) %>% 
  ggplot() +
  geom_line(aes(x=day_num, y=perc_20_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  geom_line(aes(x=day_num, y=average_residual), size=2, color='darkgreen') +
  geom_line(aes(x=day_num, y=perc_80_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  ggtitle("Residuals", subtitle='Differential Equation ML Beta') +
  labs(y = "Absolute Residual", x="Day", fill='Anruf Empfehlung') +
  theme(text = element_text(size = 12), legend.position = "right")


## Diff_Eq_Last_Beta: Visualize average absolute residuals per day:
df_forecasts %>% 
  group_by(day_num) %>% 
  summarise(
    perc_20_quantile_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.2),
    average_residual = mean(residuals_Diff_Eq_Last_Beta),
    perc_80_quantile_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.8),
  ) %>% 
  ggplot() +
  geom_line(aes(x=day_num, y=perc_20_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  geom_line(aes(x=day_num, y=average_residual), size=2, color='darkgreen') +
  geom_line(aes(x=day_num, y=perc_80_quantile_residual), linetype='dashed', size=1, color='darkgreen') +
  ggtitle("Residuals", subtitle='Differential Equation Last Beta') +
  labs(y = "Absolute Residual", x="Day", fill='Anruf Empfehlung') +
  theme(text = element_text(size = 12), legend.position = "right")



#### 3) Model Performance Analysis ####

### By Calendar Week
## MAPE:
df_metrics %>% group_by(calendar_week_start_forecast) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>%
  mutate(
    calendar_week_start_forecast = as.character(calendar_week_start_forecast)
  ) %>% 
  pivot_longer(!calendar_week_start_forecast, names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima')
  ) %>% 
  ggplot(aes(x=calendar_week_start_forecast, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  ggtitle("Forecasting Model Evaluation") +
  labs(y = "Mode of MAPE (Mean Absolute Percentage Error)", x="Calendar Week", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw()
  
  
## RMSE:
df_metrics %>% group_by(calendar_week_start_forecast) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-rmse`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-rmse`, 0.5),
    sarima = quantile(`Sarima-rmse`, 0.5),
    ensemble = quantile(`Ensemble-rmse`, 0.5),
  ) %>%
  mutate(
    calendar_week_start_forecast = as.character(calendar_week_start_forecast)
  ) %>% 
  pivot_longer(!calendar_week_start_forecast, names_to='model', values_to = 'rmse') %>%  
  filter(
    model %in% c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima')
  ) %>% 
  ggplot(aes(x=calendar_week_start_forecast, y=rmse, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  ggtitle("Forecasting Model Evaluation") +
  labs(y = "Mode of RMSE (Root Mean Squared Error)", x="Calendar Week", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw()


### By Calendar Week
## MAPE:
df_metrics %>%
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>%
  pivot_longer(c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima', 'ensemble'), 
               names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima')
  ) %>% 
  ggplot(aes(x=model, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  ggtitle("Forecasting Model Evaluation") +
  labs(y = "Mode of MAPE (Mean Absolute Percentage Error)", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw()

## RMSE:
df_metrics %>%
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-rmse`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-rmse`, 0.5),
    sarima = quantile(`Sarima-rmse`, 0.5),
    ensemble = quantile(`Ensemble-rmse`, 0.5),
  ) %>%
  pivot_longer(c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima', 'ensemble'), 
               names_to='model', values_to = 'rmse') %>%  
  filter(
    model %in% c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima')
  ) %>% 
  ggplot(aes(x=model, y=rmse, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  ggtitle("Forecasting Model Evaluation") +
  labs(y = "Mode of RMSE (Root Mean Squared Error)", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw()

