library(tidyverse)
library(caret)
library(GGally)
library(ggfortify)
library(scales)
library(RColorBrewer)
library(lubridate)


setwd("/Users/heidemann/documents/private/Project_Seminar/Backend")


col_diffeq1 <- '#006450'
col_diffeq2 <- '#00b0f6'
col_arima <- '#e76bf3'
col_ensemble <- '#e6a902'



#### 1) Import Predictions ####

# Produced by Model Evaluation Pipeline. This data is used to evaluate the different approaches.
# The Forecasts Dataframe contains detailed information regarding the forecasts of the different
# models and the correct data (validation data). Using the idx column it this table is connected
# to the Metrics Dataframe that evaluates the different approaches over each forecasting horizon 
# and district combination.

df_forecasts <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_forecasts_02_02_03:38.csv")
df_metrics <- read_csv(file="../Assets/Data/Evaluation/model_validation_data_metrics_02_02_03:38.csv")

## Preprocessing:
# Rename First Column:
df_forecasts <- df_forecasts %>%
  # Add day column:
  rename(
    day_num = names(df_forecasts)[1]
  ) %>% 
  mutate(
    day_num = day_num + 1
  )

# Add month column:
df_metrics <- df_metrics %>% 
  mutate(
    month = format(as.POSIXlt(start_day_val, format="%d/%m/%Y"), "%Y-%m"),
    date = format(as.POSIXlt(start_day_val, format="%d/%m/%Y"), "%d-%m-%Y")
  )


#### 2) Residuals Analysis ####

## Compute relative residuals:
df_forecasts <- df_forecasts %>% mutate(
  rel_residuals_Diff_Eq_Last_Beta = residuals_Diff_Eq_Last_Beta / y_val,
  rel_residuals_Diff_Eq_Ml_Beta = residuals_Diff_Eq_ML_Beta / y_val,
  rel_residuals_Sarima = residuals_Sarima / y_val,
  rel_residuals_ensemble = residuals_Ensemble / y_val,
)

### Differentiate forecasts into three different groups based on average incidence for y_val:
df_forecasts %>% group_by(idx) %>% 
  summarize(
    avg_infections = mean(y_val)
  ) %>% 
  mutate(
    quantile25_infections = quantile(avg_infections, 0.25),
    quantile50_infections = quantile(avg_infections, 0.50),
    quantile75_infections = quantile(avg_infections, 0.75),
  )


weekly_district_infections <- df_forecasts %>% group_by(idx) %>% 
  summarize(avg_infections = mean(y_val))

groups_vector <- cut(weekly_district_infections$avg_infections,
                     breaks = c(-Inf, 100, 200, Inf),
                     labels= c(100,200,99999))

weekly_district_infections <- weekly_district_infections %>% 
  mutate(
    class = groups_vector
  )

df_joined <- inner_join(df_forecasts, weekly_district_infections, by='idx')


# Set 90% quantile for upper bound and 10% quantile for lower bound.
upper_bound = 0.90
lower_bound = 0.10


## Compute historical relative errors.
# These errors are aggregated over the different models for the three different classes.
# Here the idea is that forecasting residuals increase with the forecasting date. Forecasting 
# for the next day is easier and therefore the residuals are expected to be lower compared to 
# forecasting the indicence on forecasting day 14.
# Also we expected the relative residuals to be depend on how high the number of current
# infections is. We noticed that our models had a hard time forecasting when the current incidences
# were very low, which might be due to a higher signal-to-noise ratio in such scenarios.
# To not inflate our prediction intervals we therefore compute for each forecasting day and each
# model the quantiles for three different groups depending on how high the indicences were around that time.

# Group by class
df_prediction_intervals_perc <- df_joined %>% 
  group_by(class, day_num) %>% 
  summarise(
    # Diff Eq Last Beta
    upper_perc_diff_eq_last_beta = quantile(rel_residuals_Diff_Eq_Last_Beta, upper_bound),
    lower_perc_diff_eq_last_beta  = quantile(rel_residuals_Diff_Eq_Last_Beta, lower_bound),
    
    # Diff Eq Ml Beta
    upper_perc_diff_eq_ml_beta  = quantile(rel_residuals_Diff_Eq_Ml_Beta, upper_bound),
    lower_perc_diff_eq_ml_beta  = quantile(rel_residuals_Diff_Eq_Ml_Beta, lower_bound)
  ) 

## The above computed relative residual quantiles are used for computing the prediction intervals
## for both differential equation models. The CSV file that is exported below is integrated into
## the prediction pipeline.

# Export intervals:
write_csv(df_prediction_intervals_perc, file="../Assets/Forecasts/PredictionIntervals/prediction_intervals.csv")

#### 2b) Visualization of Absolute Residuals ####
### Last_Beta
## Diff_Eq_Last_Beta: Visualize residuals on forecasting day 14:
df_forecasts %>% 
  filter(day_num == 14) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_Last_Beta), 
                 binwidth = 2) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-150,150))

## Diff_Eq_Last_Beta: Visualize residuals on forecasting day 1:
df_forecasts %>% 
  filter(day_num == 1) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_Last_Beta), 
                 binwidth = 1) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-50,50))

### ML-Beta
## Diff_ML_Last_Beta: Visualize residuals on forecasting day 14:
df_forecasts %>% 
  filter(day_num == 14) %>% 
  ggplot() +
  geom_histogram(aes(x=residuals_Diff_Eq_ML_Beta), 
                 binwidth = 2) +
  xlim(-1000,1000) +
  coord_cartesian(xlim =c(-150,150))

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
    average_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.5),
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
    average_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.5),
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
    average_residual = quantile(residuals_Diff_Eq_Last_Beta, 0.5),
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
  
  
## RMSE: - per week
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

## RMSE: - per district
df_metrics %>% group_by(district) %>%
  filter(district %in% sample(df_metrics$district, 40, replace=FALSE)) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-rmse`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-rmse`, 0.5),
    sarima = quantile(`Sarima-rmse`, 0.5),
    ensemble = quantile(`Ensemble-rmse`, 0.5),
  ) %>%
  pivot_longer(!district, names_to='model', values_to = 'rmse') %>%  
  filter(
    model %in% c('diff_eq_last_beta', 'diff_eq_ml_beta', 'sarima')
  ) %>% 
  ggplot(aes(x=district, y=rmse, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  ggtitle("Forecasting Model Evaluation") +
  labs(y = "Mode of RMSE (Root Mean Squared Error)", x="District", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))





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



#### Overall performance: ####
df_metrics %>%
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>%
  rename(
    'SEIURV Last Beta' = diff_eq_last_beta,
    'SEIURV ML Beta' = diff_eq_ml_beta,
    'ARIMA' = sarima,
    'Ensemble' = ensemble,
  ) %>% 
  pivot_longer(c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'), 
               names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'),
  ) %>%
  ggplot(aes(x=model, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black', size=0.7) +
  # scale_fill_manual(values=c("#043B05", "#1A4B75", "#A62189")) +
  scale_fill_manual(values=c(col_arima, col_ensemble, col_diffeq1, col_diffeq2)) +
  ggtitle("Forecasting Evaluation", subtitle = 'Overall Model Performance in 2021') +
  labs(x = '') +
  labs(y = "Average MAPE", fill='Model') +
  theme_bw() +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#### Monthly performance: ####
df_metrics %>% group_by(month) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>%
  rename(
    'SEIURV Last Beta' = diff_eq_last_beta,
    'SEIURV ML Beta' = diff_eq_ml_beta,
    'ARIMA' = sarima,
    'Ensemble' = ensemble,
  ) %>% 
  pivot_longer(c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'), 
               names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'),
    month %in% c('2021-09', '2021-10', '2021-11', '2021-12', '2022-01')
  ) %>%
  ggplot(aes(x=month, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  # scale_fill_manual(values=c("#043B05", "#1A4B75", "#A62189")) +
  scale_fill_manual(values=c(col_arima, col_ensemble, col_diffeq1, col_diffeq2)) +
  ggtitle("Forecasting Model Evaluation") +
  labs(x = '') +
  labs(y = "Average MAPE", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))



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



#### Monthly performance:
df_metrics %>% group_by(month) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>%
  rename(
    'SEIURV Last Beta' = diff_eq_last_beta,
    'SEIURV ML Beta' = diff_eq_ml_beta,
    'ARIMA' = sarima,
    'Ensemble' = ensemble,
  ) %>% 
  pivot_longer(c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'), 
               names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'),
    month %in% c('2021-09', '2021-10', '2021-11', '2021-12', '2022-01')
  ) %>%
  ggplot(aes(x=month, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  # scale_fill_manual(values=c("#043B05", "#1A4B75", "#A62189")) +
  scale_fill_manual(values=c(col_arima, col_ensemble, col_diffeq1, col_diffeq2)) +
  ggtitle("Forecasting Model Evaluation") +
  labs(x = '') +
  labs(y = "Average MAPE", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#### Performance over Time: ####
df_metrics %>% group_by(start_day_val) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
    date = date
  ) %>%
  rename(
    'SEIURV Last Beta' = diff_eq_last_beta,
    'SEIURV ML Beta' = diff_eq_ml_beta,
    'ARIMA' = sarima,
    'Ensemble' = ensemble,
  ) %>% 
  pivot_longer(c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'), 
               names_to='model', values_to = 'mape') %>%  
  filter(
    model %in% c('SEIURV ML Beta', 'ARIMA'),
    date %in% c('18-11-2021', '25-11-2021' , '09-12-2021')
  ) %>%
  rename(
    start_forecasting = date
  ) %>% 
  ggplot(aes(x=start_forecasting, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  facet_wrap(~model, scales = "fixed") +
  scale_fill_manual(values=c(col_arima, col_diffeq2)) +
  ggtitle("Forecasting Evaluation", subtitle='Evaluation for Different Time Periods') +
  labs(x = '') +
  labs(y = "Average MAPE", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  
  
#### Performance per District: ####
df_metrics %>% group_by(district) %>% 
  summarize(
    diff_eq_last_beta = quantile(`Diff_Eq_Last_Beta-mape`, 0.5),
    diff_eq_ml_beta = quantile(`Diff_Eq_ML_Beta-mape`, 0.5),
    sarima = quantile(`Sarima-mape`, 0.5),
    ensemble = quantile(`Ensemble-mape`, 0.5),
  ) %>% 
  filter(
    district %in% c('Münster','Dortmund','Aachen','München, Kreis','Borken','Ortnaukreis','Dresden','Erzgebirgskreis','Fulda','Hannover')
  ) %>% 
  rename(
    District = district,
    'SEIURV Last Beta' = diff_eq_last_beta,
    'SEIURV ML Beta' = diff_eq_ml_beta,
    'ARIMA' = sarima,
    'Ensemble' = ensemble,
  ) %>% 
  pivot_longer(c('SEIURV Last Beta', 'SEIURV ML Beta', 'ARIMA', 'Ensemble'), 
               names_to='model', values_to = 'mape') %>%
  filter(model %in% c('SEIURV Last Beta', 'Ensemble')) %>% 
  ggplot(aes(x=District, y=mape, fill=model)) +
  geom_bar(position='dodge', stat='identity', color='black') +
  # scale_fill_manual(values=c(col_arima, col_ensemble, col_diffeq1, col_diffeq2)) +
  # scale_fill_manual(values=c(col_arima, col_diffeq2)) +
  scale_fill_manual(values=c(col_ensemble, col_diffeq1)) +
  facet_wrap(~model, scales = "fixed") +
  ggtitle("Forecasting Evaluation", subtitle='Evaluation for Different Districts') +
  labs(x = '') +
  labs(y = "Average MAPE", fill='Model') +
  theme(text = element_text(size = 12), legend.position = "right") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))




#### 4) SEIR Beta Plot ####
# Plot used as a hypothetical example for the final presentation.

my_fun <- function(x, a){
  val <- 99 + exp(a*(x-1))
  return (val)
}


days <- seq(1, 21, by=1)

low_beta <- sapply(days, my_fun, a=0.17)
medium_beta <- sapply(days, my_fun, a=0.2)
high_beta <- sapply(days, my_fun, a=0.23)

tibble(days, low_beta, medium_beta, high_beta) %>%
  pivot_longer(c('low_beta','medium_beta','high_beta'), 
               names_to='beta', values_to = 'infections') %>% 
  rename(
    day = days
  ) %>% 
  filter(
    # beta %in% c('low_beta')
    beta %in% c('low_beta','medium_beta')
    # beta %in% c('low_beta','medium_beta','high_beta')
  ) %>% 
  ggplot(aes(x=day, y=infections, fill=beta, color=beta)) +
  geom_line(size=1) +
  # scale_color_manual(values=c("#2a5906")) +
  scale_color_manual(values=c("#2a5906", "#e6a902")) +
  # scale_color_manual(values=c("#a11b06", "#2a5906", "#e6a902")) +
  ggtitle("Effect of Changing Beta") +
  labs(x = "Days", y="Daily Infections", fill='Beta', color='Beta') +
  theme_bw() +
  theme(text = element_text(size = 13)) +
  coord_cartesian(ylim = c(100,200))
