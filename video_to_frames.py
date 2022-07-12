import argparse
import os
import shutil 
import cv2
import numpy as np
from os import path

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_all_video_frames(path, max_frames=0):
  cap = cv2.VideoCapture(path)
  frames = []  
  try:
    while True:
      #frame = numpy array
      ret, frame = cap.read()
      if not ret:
        break        
      
      frames.append(frame)    
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) #/ 255.0


def get_start_indices(num_frames, frames_per_segment, num_segments, uniform=False) :
    """
    For each segment, choose a start index from where frames
    are to be loaded from.
    Args:
        record: VideoRecord denoting a video sample.
    Returns:
        List of indices of where the frames of each
        segment are to be loaded from.
    """
    # choose start indices that are perfectly evenly spread across the video frames.
    if uniform:
        distance_between_indices = (num_frames - frames_per_segment + 1) / float(num_segments)        
        frame_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                  for x in range(num_segments)])
    # randomly sample start indices that are approximately evenly spread across the video frames.
    else:
        max_valid_start_index = (num_frames - frames_per_segment + 1) // num_segments

        frame_indices = np.multiply(list(range(num_segments)), max_valid_start_index) + \
                  np.random.randint(max_valid_start_index, size=num_segments)

    return frame_indices



def export_frames(video, frames_out_path, frames_per_segment, frame_start_indices: 'np.ndarray[int]', resize=(384,384)):

    frame_start_indices = frame_start_indices
    images = list()

    # from each start_index, load self.frames_per_segment
    # consecutive frames
    count = 1
    for start_index in frame_start_indices:

        frame_index = int(start_index)

        # load frames_per_segment consecutive frames
        for _ in range(frames_per_segment):
            image = video[frame_index]
            image = crop_center_square(image)     
            image = cv2.resize(image, resize)       
              
            out =  os.path.join(frames_out_path, "frame_"+str(count)+ "_" + str(frame_index)+".jpg")
            
            cv2.imwrite(out, image)
            if frame_index < video.shape[0]:
                frame_index += 1
                count += 1


def main(origin_path, num_segments, frames_per_segment, uniform=False):
      for categories, subdirs, files in os.walk(origin_path): # "/content/drive/MyDrive/train_vid"   
        for vid_name in files:        
            
            if (".mp" in vid_name):  
              name_no_suffix = vid_name.replace(".mp4", "")
              name_no_suffix = name_no_suffix.replace(".mp3", "")
              frames_path = os.path.join(categories, "frames_" + name_no_suffix)
              vido_path =  os.path.join(categories, vid_name)
              if (path.exists(frames_path)):
                 shutil.rmtree(frames_path)
              if(not path.exists(frames_path)):          
                os.mkdir(frames_path)
              
              video = load_all_video_frames(vido_path)
              num_frames = video.shape[0]              
              frame_start_indices = get_start_indices(num_frames, frames_per_segment, num_segments,uniform)                
              export_frames(video, frames_path, frames_per_segment, frame_start_indices, resize=(384,384))





if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--origin_path', type=str, default=None, help='Path to all the videos')
    p.add_argument('--num_segments', type=int, help='Number of segments')
    p.add_argument('--frames_per_segment', type=int, required=True, help='number of consecutive frames per segment.')   
    p.add_argument('--uniform', type=bool, default=False, required=False, help='uniform sampling.')  
    main(**vars(p.parse_args()))
