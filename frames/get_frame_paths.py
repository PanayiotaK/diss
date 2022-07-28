import os
import csv
import pandas as pd
import numpy as np
import argparse


def main(origin_path,out_file_name):
  partinioned = origin_path.split("original/",1)[1]
  out_file = os.path.join(origin_path, partinioned+out_file_name) 
  with open(out_file, 'wt', encoding='utf-8') as f:  
    tsv_writer = csv.writer(f, delimiter='\t')     
    for categories, subdirs, files in os.walk(origin_path): # "/content/drive/MyDrive/train_vid" 
          # print(subdir)  
          for subdir in subdirs:
            if "frames_" in subdir:            
              p = os.path.join(categories, subdir)
              listOfFile = os.listdir(p)              
              for entry in listOfFile:            
                fullPath = os.path.join(p, entry)                
                tsv_writer.writerow([fullPath])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--origin_path', type=str, default=None, help='Path to all the videos')   
    p.add_argument('--out_file_name', type=str, default='data_paths.tsv', help='Name of the tsv with all the frame paths will be saved' )
    
    main(**vars(p.parse_args()))