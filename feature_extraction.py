import os
import cv2
import numpy as np
import pandas as pd

data_folder = "data"
categories = ["water", "desert", "cloudy", "green_area"]
all_features = []
print("Starting...")

for category in categories:
    #create path
    folder_path = os.path.join(data_folder, category)
    image_filenames = os.listdir(folder_path)

    for image_name in image_filenames:
        image_path = os.path.join(folder_path, image_name)

        #load image
        img = cv2.imread(image_path)
        #convert to RGB from cv2 BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #get avg values for every color
        avg_r = np.mean(img_rgb[:,:,0])
        avg_g= np.mean(img_rgb[:,:,1])
        avg_b= np.mean(img_rgb[:,:,2])

        #get standard deviatons of colors
        std_r = np.std(img_rgb[:,:,0])
        std_g= np.std(img_rgb[:,:,1])
        std_b= np.std(img_rgb[:,:,2])

        #edge detection
        edges = cv2.Canny(img_rgb,100,200)
        edgesum = np.sum(edges > 0)

        
        features = {
            "image_name": image_name,
            "avg_r": avg_r,
            "avg_g": avg_g,
            "avg_b": avg_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
            "edge_sum": edgesum,
            "label": category
        }
        all_features.append(features)

#create dataframe
df = pd.DataFrame(all_features)
df.to_csv("features_data.csv")
print("features_data.csv created")