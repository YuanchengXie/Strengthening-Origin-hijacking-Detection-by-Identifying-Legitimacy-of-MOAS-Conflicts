from BML import data
from BML import utils
from BML import transform
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import json

# standardized time range
def standardized_time_range(time, duration_before=8, duration_after=2):
    # Convert string time to a datatime object
    specific_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    # Calculate the starting and ending points of the time range
    start_time = str(specific_time - timedelta(minutes=duration_before))
    end_time = str(specific_time + timedelta(minutes=duration_after))

    return start_time, end_time

# splitting time into integers
def split_time(time):
    year_month_day = time.split(' ')[0]
    hour_minute_second = time.split(' ')[1]
    year = int(year_month_day.split('-')[0])
    month = int(year_month_day.split('-')[1])
    day = int(year_month_day.split('-')[2])
    hour = int(hour_minute_second.split(':')[0])
    minute = int(hour_minute_second.split(':')[1])
    second = int(hour_minute_second.split(':')[2])
    return [year, month, day, hour, minute, second]

# Extract hijacking event data
def extract_event():

    # event dataset
    event = []
    file_data = pd.read_json('/home/ycxie/xyc/MOAS/venv/features_dataset.json')
    # traverse data
    for index, row in file_data.iterrows():
        prefix = row['hijacker_prefix']
        asn = str(row['hijacker_ASN'])
        label = row['label']
        label_prefix_asn = str(label) + '_' + prefix + '_' + asn

        filename = prefix.split('/')[0] + '_' + prefix.split('/')[1] + '_' + asn + '_' + str(label)
        start_time = str(row['hijack_start_time'])
        hijack_start_time, hijack_end_time = standardized_time_range(start_time,duration_before=15, duration_after=15)
        time_list = split_time(hijack_start_time) + split_time(hijack_end_time)
        event.append((filename,label,time_list))
    return event

def calculate_characteristics(filename,label,time):
    if label == 1:
        print_label = 'exact hijacke'
    elif label == 2:
        print_label = 'sub-prefix hijack'
    elif label == 3:
        print_label = 'exact legitimate MOAS'
    else:
        print_label = 'sub-prefix legitimate MOAS'
    print('task ' + filename + ' is beginning')
    print('Task type: ' + print_label)
    task_label = 'training_set'
    # Data collection
    folder = "dataset_features_train/"    # the storage address of feature data
    dataset = data.Dataset(folder)
    dataset.setParams({
        "PrimingPeriod": 1*60, # 1 hours of priming data
        "Collectors": ["rrc00",'route-views2'], # Collect data from rrc00 and route-views2
    })

    # Extract the time element when an event occurs
    startyear, startmonth, startday, starthour, startminute, startsecond, \
    endyear, endmonth, endday, endhour, endminute, endsecond = time

    # Set the data storage address and data collection time for events
    dataset.setPeriodsOfInterests([
        {
        "name": filename,
        "label": task_label,
        "start_time": utils.getTimestamp(startyear, startmonth, startday, starthour, startminute, startsecond),  # for example: August 25, 2017, 3:00 UTC
        "end_time": utils.getTimestamp(endyear, endmonth, endday, endhour, endminute, endsecond)  # for example: August 25, 2017, 4:00 UTC
        }
    ])

    # run the data collection
    utils.runJobs(dataset.getJobs(), folder+"collect_jobs")

    # statistic features extraction every minute
    S_datTran = transform.DatasetTransformation(folder, "BML.transform", "Features")
    S_datTran.setParams({
            "global":{
                "Period": 1,
            }
    })

    # run the data transformation
    utils.runJobs(S_datTran.getJobs(), folder+"S_transform_jobs")

    # graph features extraction every minute
    G_datTran = transform.DatasetTransformation(folder, "BML.transform", "GraphFeatures")
    G_datTran.setParams({
            "global":{
                "Period": 1,
            }
        })

    # run the data transformation
    utils.runJobs(G_datTran.getJobs(), folder+"G_transform_jobs")

    # hijacker features extraction every minute
    H_datTran = transform.DatasetTransformation(folder, 'computing_hijacker_feature', "Hijacker_features")
    H_datTran.setParams({
        "global": {
            "Period": 1,
        }
    })

    # run the data transformation
    utils.runJobs(H_datTran.getJobs(), folder + "H_transform_jobs")

    print(print_label + filename + ' is finished!')

if __name__ == '__main__':
    event = extract_event()

    # i = 0
    # for filename, label,timelist in event:
    #     print('Processing the'+str(i)+'\\'+str(len(event))+'个任务')
    #     i += 1
    #     calculate_characteristics(filename,label,timelist)
    # print('task has finished!')


    #Using multithreading to execute a task list
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(calculate_characteristics, filename, label, time_list)
                   for filename, label, time_list in event]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

