import os
from PIL import Image
import numpy as np

path = 'data/training_dataset/motobike'
out_path = 'data/filter/motorbike'

os.system(f'mkdir -p {out_path}')
for x in os.listdir(path):
    try:
        img = Image.open(os.path.join(path, x))
        img_g = img.convert('L')
        # img_np = np.array(img_g)
        # ratio = np.sum(img_np > 250) / np.product(img_np.shape)
        os.system(f'cp {path}/{x} {out_path}/{x}')
    except KeyboardInterrupt:
        break
    except:
        continue
