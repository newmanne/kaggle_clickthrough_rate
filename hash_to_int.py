import os
from collections import Counter
import numpy as np
import pandas as pd

COLS_TO_TRANLSATE = "site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,C1,C14,C15,C16,C17,C18,C19,C20,C21".split(",") 

COLS = "id,click,day,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")

TRAIN_DATA = "../train.csv"
TEST_DATA = "../test.csv"
TRAIN_MERGED_DATA = '../merged_train.csv'
TEST_MERGED_DATA = "../merged_test.csv"
COUNTERS_FILE = '../counters.txt'

CHUNK_SIZE = 1000000
THRESH = 100

def create_counters():    
    counters = {}
    for col in COLS_TO_TRANLSATE:
        counters[col] = Counter()
    return counters
    
def update_counter(path,counters):
    df = pd.read_csv(path,chunksize=CHUNK_SIZE,iterator=True)
    for chunk in df:    
        for col in COLS_TO_TRANLSATE:
            counters[col].update(chunk.ix[:,col])
        print chunk.id.max()

def convert_counts_to_id(counters):
    ids = {}
    for col in COLS_TO_TRANLSATE:
        ids[col] = {}
        imax = 0
        highest_seen = 0
        for i,(val,count) in enumerate(counters[col].most_common()):
            if imax == 0 and count <= THRESH:
                imax = i
            index = i if count > THRESH else imax
            highest_seen = max(index, highest_seen)
            ids[col][val] = index
        print "Col " + col + " highest seen is " + str(highest_seen)
    return ids
    
def write_translated(input_path,output_path,ids,mode="w",start_id=0):
    df = pd.read_csv(input_path,chunksize=CHUNK_SIZE,iterator=True)
    for i, chunk in enumerate(df):
        for col in COLS_TO_TRANLSATE:
            chunk.ix[:,col] = chunk.ix[:,col].map(ids[col])
        chunk["id"] = np.arange(chunk.shape[0]) + i*CHUNK_SIZE + 1 + start_id
        chunk["day"] = chunk["hour"].map(lambda v: int(str(v)[-4:-2]))
        chunk["hour"] = chunk["hour"].map(lambda v: int(str(v)[-2:]))
        
        if "click" not in chunk.columns:
            chunk["click"] = 0
        chunk = chunk.ix[:,COLS]

        if i == 0 and mode == "w":
            chunk.to_csv(output_path,index=False)
        else:            
            chunk.to_csv(output_path,index=False,mode="a",header=False)

        print chunk.id.max()

    return chunk.id.max()

def write_counters(file_path, counters):
    with open(file_path, 'w') as outfile:
        for counter in counters.values():
            outfile.write('%s\n' % (counter))

if __name__ == "__main__":
    counters = create_counters()
    update_counter(TRAIN_DATA,counters)
    update_counter(TEST_DATA,counters)
    write_counters(COUNTERS_FILE, counters)
    ids = convert_counts_to_id(counters)
    max_id = write_translated(TRAIN_DATA,TRAIN_MERGED_DATA,ids)    
    _ = write_translated(TRAIN_DATA,TEST_MERGED_DATA,ids, start_id=max_id)
    max_id = write_translated(TEST_DATA,TEST_MERGED_DATA,ids)
    _ = write_translated(TEST_DATA,TEST_MERGED_DATA,ids, start_id=max_id)