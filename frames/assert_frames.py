import pandas as pd
import os 
from os import path
import argparse

def main(origin_path):

    all_vids = []
    video_with_frames = []
    no_frames = []
    right_no = []
    
    for categories, subdirs, files in os.walk(origin_path): # "/content/drive/MyDrive/train_vid"   
        for vid_name in files:  
            if (".mp" in vid_name):            
                name_no_suffix = vid_name.replace(".mp4", "")
                name_no_suffix = name_no_suffix.replace(".mp3", "")
                all_vids.append(name_no_suffix)
                frames_path = os.path.join(categories, "frames_" + name_no_suffix)
                
                if (path.exists(frames_path)):
                    video_with_frames.append(name_no_suffix)
                    frames = os.listdir(frames_path)
                    # if len(frames) == 50:
                    no_frames.append(len(frames))
                    if len(frames) != 50 :
                        right_no.append('Wrong no')
                    else:
                        right_no.append('')

    df_all_vid = pd.DataFrame()
    df_all_vid['vid_name'] = all_vids
    
    df_video_frames = pd.DataFrame()
    df_video_frames['name'] = video_with_frames
    df_video_frames['no_frames'] = no_frames
    df_video_frames['right_no'] = right_no

    df_video_frames.to_csv(os.path.join(origin_path,'has_frames.csv'))
    df_all_vid.to_csv(os.path.join(origin_path,'all_vids.csv'))
            
 



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--origin_path', type=str, default=None, help='Path to all the videos')   
    main(**vars(p.parse_args()))
