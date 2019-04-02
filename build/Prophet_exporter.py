#!/usr/bin/env python
# coding: utf-8

# In[12]:


import requests
import prometheus
import json
import os
import pandas
from datetime import datetime, timedelta
from sortedcontainers import SortedDict
from fbprophet import Prophet
from prometheus_client import CollectorRegistry, generate_latest, REGISTRY, Counter, Gauge, Histogram

def get_df_from_json(metric, metric_dict_pd={}, data_window=5):
    '''
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    '''
    current_time = datetime.now()
    earliest_data_time = current_time - timedelta(days = data_window)


    print("Pre-processing Data...........")
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(SortedDict(row['metric']))[11:-1] # Sort the dictionary and then convert it to string so it can be hashed
        # print(metric_metadata)
        # print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            metric_dict_pd[metric_metadata]['ds'] = pandas.to_datetime(metric_dict_pd[metric_metadata]['ds'], unit='s')
            pass
        else:
            temp_df = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            temp_df['ds'] = pandas.to_datetime(temp_df['ds'], unit='s')
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].append(temp_df, ignore_index=True)
            mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time)
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
            pass
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].dropna()
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].drop_duplicates('ds').sort_values(by=['ds']).reset_index(drop = True)

        if len(metric_dict_pd[metric_metadata]) == 0:
            del metric_dict_pd[metric_metadata]
            pass
        pass

    return metric_dict_pd


def predict_metrics(pd_dict, prediction_range=1):
    '''
    This Function takes input a dictionary of Pandas DataFrames, trains the Prophet model for each dataframe and returns a dictionary of predictions.
    '''

    total_label_num = len(pd_dict)
    # LABEL_LIMIT = limit_labels
    PREDICT_DURATION = prediction_range

    current_label_num = 0
    limit_iterator_num = 0

    predictions_dict = {}

    for meta_data in pd_dict:
        try:
            current_label_num += 1
            limit_iterator_num += 1

            print("Training Label {}/{}".format(current_label_num,total_label_num))
            data = pd_dict[meta_data]

            print("----------------------------------\n")
            print(meta_data)
            print("Number of Data Points: {}".format(len(pd_dict[meta_data])))
            print("----------------------------------\n")

            data['ds'] = pandas.to_datetime(data['ds'], unit='s')

            train_frame = data

            # Prophet Modelling begins here
            m = Prophet(daily_seasonality = True, weekly_seasonality=True)

            print("Fitting the train_frame")
            m.fit(train_frame)

            future = m.make_future_dataframe(periods=int(PREDICT_DURATION),freq="1MIN")

            forecast = m.predict(future)

            # To Plot
            fig1 = m.plot(forecast)
            #
            fig2 = m.plot_components(forecast)
            forecast['timestamp'] = forecast['ds']
            forecast = forecast[['timestamp','yhat','yhat_lower','yhat_upper']]
            forecast = forecast.set_index('timestamp')

            # Store predictions in output dictionary
            predictions_dict[meta_data] = forecast

            # forecast.plot()
            # plt.legend()
            # plt.show()
        except ValueError as exception:
            if str(exception) == "ValueError: Dataframe has less than 2 non-NaN rows.":
                print("Too many NaN values........Skipping this label")
                limit_iterator_num -= 1
            else:
                raise exception
        pass

    return predictions_dict

url = 'http://61.28.251.119:9090'
metric_name = 'mem_used'
print("Get data of metric {}".format(metric_name))
data_window = 3
# Chunk size, download the complete data, but in smaller chunks, should be less than or equal to DATA_SIZE
chunk_size = str(os.getenv('CHUNK_SIZE','2h'))

# Net data size to scrape from prometheus
data_size = str(os.getenv('DATA_SIZE','5h'))

train_schedule = int(os.getenv('TRAINING_REPEAT_HOURS',6))

TRUE_LIST = ["True", "true", "1", "y"]

prom = prometheus.Prometheus(url,chunk_size, data_size)
metric = prom.get_metric(metric_name)
print("metric collected.")
metric = json.loads(metric)
data_dict = {}
data_dict = get_df_from_json(metric, data_dict, data_window)
#print(data_dict)
predictions_dict_prophet = {}
predictions_dict_fourier = {}
    


# In[2]:



predictions_dict_prophet = predict_metrics(data_dict)


# In[3]:


from ast import literal_eval

single_label_data_dict = {}
existing_config_list = list(data_dict.keys())
for existing_config in existing_config_list:
    single_label_data_dict[existing_config] = data_dict[existing_config]

                    
current_metric_metadata = list(single_label_data_dict.keys())[0]
current_metric_metadata_dict = literal_eval(current_metric_metadata)
print(current_metric_metadata)


# In[4]:



from prometheus_client import Gauge        
registry = CollectorRegistry()                   
predicted_metric_name = "predicted_" + metric_name
PREDICTED_VALUES_PROPHET = Gauge(predicted_metric_name + '_prophet', 'Forecasted value from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_PROPHET_UPPER = Gauge(predicted_metric_name + '_prophet_yhat_upper', 'Forecasted value upper bound from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_PROPHET_LOWER = Gauge(predicted_metric_name + '_prophet_yhat_lower', 'Forecasted value lower bound from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_ANOMALY_PROPHET = Gauge(predicted_metric_name + '_prophet_anomaly', 'Detected Anomaly using the Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
registry.register(PREDICTED_VALUES_PROPHET)
registry.register(PREDICTED_VALUES_PROPHET_LOWER)
#registry.register(PREDICTED_ANOMALY_PROPHET)
registry.register(PREDICTED_VALUES_PROPHET_UPPER)

# In[9]:


for metadata in predictions_dict_prophet:
    current_metric_metadata = metadata

    print("The current time is: ",datetime.now())
    

    current_metric_metadata_dict = literal_eval(metadata)

    temp_current_metric_metadata_dict = current_metric_metadata_dict.copy()
    del temp_current_metric_metadata_dict['__name__']
    print(temp_current_metric_metadata_dict)




from flask import Flask, render_template_string, abort, Response
import prometheus_client
def get_df_from_single_value_json(metric, metric_dict_pd={}, data_window=5):
    '''
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    '''
    # metric_dict = {}
    current_time = datetime.now()
    earliest_data_time = current_time - timedelta(days = data_window)


    print("Pre-processing Data...........")
    # metric_dict_pd = {}
    # print("Length of metric: ", len(metric))
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(SortedDict(row['metric']))[11:-1] # Sort the dictionary and then convert it to string so it can be hashed
        # print(metric_metadata)
        # print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame([row['value']], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            metric_dict_pd[metric_metadata]['ds'] = pandas.to_datetime(metric_dict_pd[metric_metadata]['ds'], unit='s')
            pass
        else:
            temp_df = pandas.DataFrame([row['value']], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            temp_df['ds'] = pandas.to_datetime(temp_df['ds'], unit='s')
            # print(temp_df.head())
            # print("Row Values: ",row['values']
            # print("Temp Head Before 5: \n",temp_df.head(5))
            # print("Head Before 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail Before 5: \n",metric_dict_pd[metric_metadata].tail(5))
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].append(temp_df, ignore_index=True)
            # print("Head 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail 5: \n",metric_dict_pd[metric_metadata].tail(5))
            mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time)
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
            # del temp_df
            pass
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].dropna()
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].drop_duplicates('ds').sort_values(by=['ds']).reset_index(drop = True)

        if len(metric_dict_pd[metric_metadata]) == 0:
            del metric_dict_pd[metric_metadata]
            pass
        pass

        # print(metric_dict_pd[metric_metadata])
        # mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time) & (metric_dict_pd[metric_metadata]['ds'] <= current_time)
        # metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
        # break
    return metric_dict_pd

live_data_dict = {}
app = Flask(__name__)

@app.route('/metrics')
def metrics():
    global predictions_dict_prophet, predictions_dict_fourier, current_metric_metadata, current_metric_metadata_dict, metric_name, url, live_data_dict

    for metadata in predictions_dict_prophet:

            #Find the index matching with the current timestamp
            index_prophet = predictions_dict_prophet[metadata].index.get_loc(datetime.now(), method='nearest')
            #index_fourier = predictions_dict_fourier[metadata].index.get_loc(datetime.now(), method='nearest')
            current_metric_metadata = metadata

            print("The current time is: ",datetime.now())
            print("The matching index for Prophet model found was: \n", predictions_dict_prophet[metadata].iloc[[index_prophet]])
           # print("The matching index for Fourier Transform found was: \n", predictions_dict_fourier[metadata].iloc[[index_fourier]])

            current_metric_metadata_dict = literal_eval(metadata)

            temp_current_metric_metadata_dict = current_metric_metadata_dict.copy()
            # delete the "__name__" key from the dictionary as we don't need it in labels (it is a non-permitted label) when serving the metrics
            del temp_current_metric_metadata_dict["__name__"]
            print("temp_metadata_dict: {}".format(temp_current_metric_metadata_dict))
            print(temp_current_metric_metadata_dict)
            # TODO: the following function does not have good error handling or retry code in case of get request failure, need to fix that
            # Get the current metric value which will be compared with the predicted value to detect an anomaly
            metric = (prometheus.Prometheus(url=url).get_current_metric_value(metric_name, temp_current_metric_metadata_dict))

            # print("metric collected.")

            # Convert data to json
            metric = json.loads(metric)

            # Convert the json to a dictionary of pandas dataframes
            live_data_dict = get_df_from_single_value_json(metric, live_data_dict)

            # Trim the live data dataframe to only 5 most recent values
            live_data_dict[metadata] = live_data_dict[metadata][-5:]
            print(live_data_dict)

            # Update the metric values for prophet model
            PREDICTED_VALUES_PROPHET.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat'][index_prophet])
            PREDICTED_VALUES_PROPHET_UPPER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat_upper'][index_prophet])
            PREDICTED_VALUES_PROPHET_LOWER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat_lower'][index_prophet])

    return Response(prometheus_client.generate_latest(registry), mimetype="text/plain")


# In[ ]:
@app.route('/')
def index():
    return '<a href="/metrics">Metrics</a>'    
    
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8888)   




# In[ ]:




