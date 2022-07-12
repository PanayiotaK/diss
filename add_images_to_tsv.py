import os
import argparse
import csv
import pandas as pd

def main(out_path, origin_path):
  with open(out_path, 'wt') as f:  
    tsv_writer = csv.writer(f, delimiter='\t') 
    for categories, subdirs, files in os.walk(origin_path): # "/content/drive/MyDrive/train_vid"   
          for subdir in subdirs:
            if "frames_" in subdir:              
              p = os.path.join(categories, subdir)
              listOfFile = os.listdir(p)            
              for entry in listOfFile:            
                fullPath = os.path.join(p, entry)
                tsv_writer.writerow([fullPath])
                

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_path', type=str, default=None, help='Path to the output tsv file -!- incluse .tsv -!-')   
    p.add_argument('--origin_path', type=str, default=None, help='Path to all the videos')   
    main(**vars(p.parse_args()))