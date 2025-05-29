''' Given a task log from android_world, dive into the episode data'''
import pickle
import os
import gzip

import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image

task = 'BrowserDraw_0'
pkl_file = os.path.expanduser(f'~/android_world/runs/run_20250522T074522593415/{task}.pkl.gz')

# Load the pickle file
with gzip.open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# For large dictionaries, look at one key at a time
for key, value in data[0].items():
    print(f"\nKey: {key}")
    print(f"Value type: {type(value)}")
    print("Value:", str(value)[:200] + "..." if len(str(value)) > 200 else value)
    input("Press Enter to continue...")  # Pause between items
    

# check the trajectory 
print(f'episode_length={data[0]["episode_length"]}')

for screen in data[0]['episode_data']['before_screenshot']:
    screenshot = np.array(screen)    
    
    # Display using matplotlib
    plt.figure(figsize=(10, 20))  # Adjust figure size as needed
    plt.imshow(screenshot)
    plt.axis('off')  # Hide axes
    plt.show()

