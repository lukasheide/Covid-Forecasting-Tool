from Backend.Data.DataManager.db_calls import get_all_table_data
from Backend.Modeling.model_validation import diff_eq_pipeline_DEPRECATED, diff_eq_pipeline_wrapper_DEPRECATED, sarima_pipeline
import pandas as pd
import matplotlib

matplotlib.interactive(True)


def main(run_diff_eq_wrapper = False, run_diff_eq_pipeline=True, run_sarima_pipeline=False):
    # Call wrapper function used for finding optimal training period length:
    # diff_eq_pipeline_wrapper()

    # Call differential equation model validation pipeline:
    end_date = '2022-01-16'
    time_frame_train_and_validation = 42
    forecasting_horizon = 14
    opendata = get_all_table_data(table_name='district_list')
    # districts = opendata['district'].tolist()

    districts = ['Münster', 'Potsdam', 'Segeberg', 'Rosenheim, Kreis', 'Hochtaunus', 'Dortmund', 'Essen', 'Bielefeld', 'Warendorf', 'Muenchen_Landeshauptstadt']

    # Call SARIMA validation pipeline:
    if run_sarima_pipeline:
        predictions = sarima_pipeline(train_end_date=end_date,
                        duration=time_frame_train_and_validation,
                        districts=districts,
                        validation_duration=forecasting_horizon,
                        visualize=True,
                        verbose=False,
                        validate=False,
                        with_db_update=False)  # should be similar to 'visualize' boolean value
        # store_results_to_db=True)

        column_name = [end_date]
        df = pd.DataFrame(predictions)
        #df_t = df.transpose()
        df.insert(0, 'district', districts, True)
        #df_t.plot(x='district', y='2021-11-15', kind='hist')
        #df_t.hist(figsize=(15,15))
        #plt.show
        #save results as csv
        df.to_csv("evaluation.csv")

    # Call wrapper function used for finding optimal training period length:
    if run_diff_eq_wrapper:
        diff_eq_pipeline_wrapper_DEPRECATED()

    # Call differential equation model validation pipeline:
    if run_diff_eq_pipeline:
        diff_eq_pipeline_DEPRECATED(train_end_date=end_date,
                                    duration=time_frame_train_and_validation,
                                    districts=districts,
                                    validation_duration=forecasting_horizon,
                                    visualize=True,
                                    verbose=False,
                                    validate=False,  # should be similar to 'visualize' boolean value
                                    store_results_to_db=False)


    ##### Stuff below will be refactored soon #####
    # end_date = 20210804
    # start_date = 20210901
    #
    # muenster_last_28_days = get_table_data(table='Essen', date1=end_date, date2=start_date,
    #                                        attributes=['date', 'seven_day_infec'], with_index=False)
    #
    # # Split into train and validation set:
    # y_train_actual = np.array(muenster_last_28_days['seven_day_infec'])[0:15]
    # y_val_actual = np.array(muenster_last_28_days['seven_day_infec'])[15:]
    #
    # # Get simulated infection cases:
    # y_train_simulation = produce_simulated_infection_counts()
    #
    # # Get starting values for compartmental model (Should come from the data pipeline later on)
    # start_vals = set_starting_values_e0_and_i0_fitted()
    #
    # # Call seirv_model pipeline:
    # pipeline_result = seirv_pipeline(y_train=y_train_actual, start_vals_fixed=start_vals)
    # y_pred = pipeline_result['y_pred_without_train_period']
    #
    # # Visualize model pipeline run:
    # plot_train_fitted_and_validation(y_train=y_train_actual, y_val=y_val_actual, y_pred=y_pred)
    #
    # # Compute metrics:
    # scores = compute_evaluation_metrics(y_pred=y_pred, y_val=y_val_actual)

    print('end reached')


if __name__ == '__main__':
    main(run_diff_eq_pipeline=False, run_sarima_pipeline=True)
