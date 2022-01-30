library(tidyverse)
library(caret)
library(GGally)
library(ggfortify)
library(scales)
library(RColorBrewer)

setwd("/Users/heidemann/documents/private/Project_Seminar/Backend")


#### 1) Import Predictions ####
df_forecasts <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_forecasts_29_01.csv")
df_metrics <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_metrics_29_01.csv")

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
) %>% 
  select(rel_residuals_Diff_Eq_Last_Beta) %>% 
  arrange(rel_residuals_Diff_Eq_Last_Beta)


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

## MAPE:
df_metrics %>% group_by(calendar_week_start_forecast) %>% 
  summarize(
    
  )
