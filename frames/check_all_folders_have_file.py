import os
import argparse
 
def check_paths(path):
  all_files = len(os.listdir(path))
  count = 0
  for file in os.listdir(path):
    remove_back = file.replace('/', '')
    check_path = os.path.join(path,file,remove_back+'data_paths.tsv')
    if os.path.exists(check_path):
      count+=1
    else:
      print("NOT HERE: ",file)
  print("should have: ", all_files, " we have:", count)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default=None, help='Path to all the videos')  
    check_paths(**vars(p.parse_args()))