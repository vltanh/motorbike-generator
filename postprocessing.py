from PIL import Image
import os
import numpy as np

path = 'output'
img_paths = [os.path.join(path, x) for x in os.listdir(path)]
will_flip = np.random.randint(0, 2, size=len(img_paths))
for i, x in enumerate(img_paths):
    if will_flip[i]:
        img = Image.open(x).transpose(Image.FLIP_LEFT_RIGHT)
        img.save(x)
