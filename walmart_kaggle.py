
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import pmdarima as pm
import time

def main():
    
    store_start = 25
    store_end = 46
    dept_start = 1
    dept_end = 100

    inp_dir = 'walmart-recruiting-store-sales-forecasting'
    inp_train = 'train.csv'
    inp_test = 'test.csv'

    df_train = pd.read_csv(inp_dir+'/'+inp_train)
    df_test = pd.read_csv(inp_dir+'/'+inp_test)
    #df_test.head()

    # store_id_picked = 1
    # depnt_id_picked = 45
    n_periods_picked = 39 # period to be predicted
    pred_start_date = '2012-11-02'

    ## store_dept that are available in train data
    tr_d, te_d = get_train_test_dict_check(df_train, df_test, 1, 1)
    #print (len(tr_d), len(te_d))

    if True:
        start_time = time.time()
        iter_fit_auto_arima(df_train, tr_d, pred_start_date, n_periods_picked, store_start, store_end, dept_start, dept_end)
        print ("Elapsed time: ", str(time.time()-start_time))

    if False:
        # prepare test df
        tmp_df_test = df_test.copy()
        tmp_df_test['Date'] = pd.to_datetime(tmp_df_test['Date'])   

        # merge with test DataFrame and put 0 for the case that no training data
        df_pred_match_test = pd.merge(tmp_df_test, out_future_forecast, on=['Date','Store','Dept'], how='left')
        df_pred_match_test = df_pred_match_test.fillna(value={'Prediction': 0})

        # checking final prediction
        con = df_pred_match_test['Store'].isin([3]) & df_pred_match_test['Dept'].isin([2])
        print(df_pred_match_test[con])



def iter_fit_auto_arima(df_train, train_dict_aval, pred_start_date, n_periods_picked, start_store, end_store, start_dept, end_dept):
    out_future_forecast = pd.DataFrame()
    f = open('model_parameters.csv', 'a')
    for i in range(start_store,end_store):
        for j in range(start_dept,end_dept):
            skip_this = False
            if j not in train_dict_aval[i]:
                skip_this = True
            
            if skip_this:
                print ('... skipping store {}, dept {}'.format(i, j))
                continue
            
            print ('doing store {}, dept {}'.format(i, j))
            df_clean_train = create_full_dataframe(df_train, i, j)
            df_clean_train = df_clean_train.drop(columns=['Store','Dept', 'IsHoliday'])
            future_forecast, best_model = fit_auto_arima(df_clean_train, 
                                                            i, 
                                                            j, 
                                                            pred_start_date, 
                                                            n_periods_picked
                                                            )

            # if (i==1) and (j==45):
            #     out_future_forecast = future_forecast.copy()
            # else:
            out_future_forecast = pd.concat([out_future_forecast, future_forecast], ignore_index=True)
            
            # save SARIMA parameters
            out_str_model = str(i) + ',' + str(j) + ',' + best_model + '\n'
            f.write(out_str_model)

    # closing csv file
    f.close()

    # save data frame
    out_name = 'prediction_store_'+str(start_store)+'_'+str(end_store)+'_dept_'+str(start_dept)+'_'+str(end_dept)
    out_future_forecast.to_csv(out_name+'.csv', index=False) 
    
    


def fit_auto_arima(df_clean_train, store_id_picked, depnt_id_picked, pred_start_date, n_periods_picked):
    try:
        # Seasonal - fit stepwise auto-ARIMA
        smodel = pm.auto_arima(df_clean_train, start_p=1, start_q=1,
                                test='adf',
                                max_p=3, max_q=3, d=None,
                                seasonal=True,
                                m=52,
                                start_P=0, 
                                start_Q=0,
                                D=1,
                                trace=False, # Whether to print status on the fits. Note that this can be very verbose
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
    except:
        smodel = 0
        print ('auto_arima not converted')

    # print summary
    if False:
        smodel.summary()
    
    # Plot diagnostic
    if False:
        smodel.plot_diagnostics(figsize=(7,5))
        plt.show()
    
    # Get Predictions
    if type(smodel) == type(0):
        future_forecast = np.zeros(n_periods_picked)
        confint = np.zeros((n_periods_picked, 2))
        best_model = 'SARIMAX(0, 0, 0, 52)'
    else:
        #future_forecast = smodel.predict(n_periods=n_periods_picked)
        future_forecast, confint = smodel.predict(n_periods=n_periods_picked, return_conf_int=True)
        best_model = smodel.summary().tables[0].data[1][1]
        
    # Create DataFrame from predictions
    index_of_fc = pd.date_range(start=pred_start_date, periods = n_periods_picked, freq='7D')
    future_forecast = pd.DataFrame(future_forecast,index = index_of_fc,columns=['Prediction'])
    
    # Plot predictions
    if False:
        pred_ci = confint
        ax = df_clean_train.plot(label='observed', figsize=(10,4))
        future_forecast['Prediction'].plot(ax=ax, label='Dynamic Forecast', alpha=.7)
        ax.fill_between(index_of_fc,
                        pred_ci[:, 0],
                        pred_ci[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('y')
        plt.legend()
        
    # Adjust DataFrame structure
    future_forecast = future_forecast.reset_index()
    future_forecast.columns = ['Date','Prediction']
    future_forecast['Store'] = store_id_picked
    future_forecast['Dept'] = depnt_id_picked
    
    return future_forecast, best_model


def create_full_dataframe(df_train, store_id, dept_id):
    df_fulldate_train = df_train[(df_train['Store'] == 1) & (df_train['Dept'] == 1)]
    df_fulldate_train = df_fulldate_train.drop(columns=['Store','Dept', 'Weekly_Sales'])
    df = df_train[(df_train['Store'] == store_id) & (df_train['Dept'] == dept_id)]
    df_left = pd.merge(df_fulldate_train, df, on='Date', how='left')
    df_left = df_left.drop(columns=['IsHoliday_y'])
    df_left.rename(columns={'IsHoliday_x':'IsHoliday'}, inplace=True)
    print (df_fulldate_train.shape)
    print (df.shape)

    #df_left = df_left.fillna(df_left.bfill())
    df_left = df_left.interpolate()
	
	# need to do another fillna becasue interpolate() cannot fill evnts before the first available
    df_left = df_left.fillna(0)
    
    df_left['Date'] = pd.to_datetime(df_left['Date'])
    df_left = df_left.set_index('Date')
    print (df_left.shape)
	
    #df_left = df_left.drop(columns=['Store','Dept', 'IsHoliday'])

    # with pd.option_context("display.max_rows", 1000): 
    #     print(df_left)
    return df_left


def get_dep_with_full_data(df_train, need_week_count=0):
    dict = {}
    a = df_train.groupby(['Store','Dept']).count().reset_index()
    for st in range(1,46):
        deps_dat_complete = []
        for de in range(1,100):
            week_cnt = 0
            df_check_num_week = a[ (a['Store']==st) & (a['Dept']==de) ]
            if len(df_check_num_week) > 0:
                week_cnt = df_check_num_week.iloc[0]['Date']
                #print (st, de, week_cnt)
        #if st > 5: break
                if need_week_count > 0:
                    if week_cnt >= need_week_count:
                        deps_dat_complete.append(de)
                else:
                    deps_dat_complete.append(de)
        dict[st] = deps_dat_complete 
    return dict

def get_train_test_dict_check(df_train, df_test, need_week_count_train, need_week_count_test):
    dict_train = get_dep_with_full_data(df_train, need_week_count=need_week_count_train)
    dict_test = get_dep_with_full_data(df_test, need_week_count=need_week_count_test)
    return (dict_train, dict_test)

def print_store_train_test_dict_check(dict_train, dict_test, store_id):
    #ps = 'with complete week_counts'
    ps = ''
    print ('Store id: {}'.format(store_id))
    print (' dict_train: expected 99 dept but there are only {} dept {}'.format(
        len(dict_train[store_id]), ps))
    print (' dict_test: expected 99 dept but there are only {} dept {}'.format(
        len(dict_test[store_id]), ps))



if __name__ == "__main__":
    main()
