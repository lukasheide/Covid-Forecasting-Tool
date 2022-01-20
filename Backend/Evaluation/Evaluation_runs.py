from Backend.Modeling.model_validation_pipeline import sarima_pipeline
import matplotlib
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import pandas as pd
from Backend.Visualization.modeling_results import plot_evaluation_metrics


def sarima_evaluation(dates, districts, forecasting_horizon):
        metrics = []
        # 7 days runs
        for i in range(len(dates)):
            end_date = dates[i]
            time_frame_train_and_validation = 28
            # Call sarima model validation pipeline:
            metrics.append(sarima_pipeline(train_end_date=end_date,
                             duration=time_frame_train_and_validation,
                             districts=districts,
                             validation_duration=forecasting_horizon,
                             visualize=False,
                             verbose=False,
                             validate=True,
                             evaluate=True))

        print(metrics)
        df = pd.DataFrame(metrics, index=dates)
        df_t = df.transpose()
        df_t.insert(0, 'district', districts, True)
        #df_t.plot(x='district', y='2021-11-15', kind='hist')
        #df_t.hist(figsize=(15,15))
        #plt.show
        compare_metrics(df_t)
        df_t.to_csv("evaluation.csv")
        print('ende')

def compare_metrics(dataframe):
    count = 0
    dates = dataframe.columns
    print(dataframe)
    for j in range(len(dates)-1):
        rmse = dataframe[dates[j+1]]
        print(rmse)
        for i in range(len(districts)):
            plot_evaluation_metrics(rmse[i], dataframe['district'], i, dates[j+1])
            count += 1

    print(rmse)

#if __name__ == '__generalization_evaluation__':
districts = ['Essen', 'Münster', 'Herne', 'Bielefeld']
#districts = ['Essen', 'Münster']
#dates = ['2020-06-20', '2021-06-20', '2021-11-15', '2021-12-08']
dates = ['2021-12-14']
#compare_metrics(districts, dates)
sarima_evaluation(dates, districts, 14)

#dates = ['2021-11-15', '2021-12-08', '2021-06-20']