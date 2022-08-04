import os 
import argparse

def remove_frames(parent):
  subfolders = [ f.path for f in os.scandir(parent) if f.is_dir() ]
  for subfile in subfolders:
    # print(subfile)    
    # sub = str(subfile)
    for frame_file in os.listdir(subfile):
      # print(frame_file)
      if  "frames_" in frame_file:
        all_frames = os.listdir(os.path.join(subfile,frame_file))              
        if len(all_frames) > 16 :
          all_frames.sort(key= lambda x: float(x.strip('frames_.jpg')))
          print(len(all_frames))
          count = 0
          for frame in all_frames:  
            count += 1   
            if count == 3:
              # print('keep path',os.path.join(parent,subfile,frame_file,frame))
              count = 0
              continue
            else:
               os.remove(os.path.join(subfile,frame_file,frame))
             
             
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=str, default=None, help='Path to all the parent video folder. Were all videos are')       
        
    remove_frames(**vars(p.parse_args()))