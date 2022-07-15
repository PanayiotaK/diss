import tensorflow as tf 
import numpy as np 

def dataset(filenames_list):
    if filenames_list :
        # initialize train dataset
        train_dataset = np.load(filenames_list[0]) 
        ds = tf.data.Dataset.from_tensor_slices((train_dataset))     
        # concatenate with the remaining files  
        for file in filenames_list[1:]: 
            read_data = np.load(file)
            add_ds = tf.data.Dataset.from_tensor_slices((read_data))
            ds = ds.concatenate(add_ds)
    else:
        print("empty list")
        

# ds