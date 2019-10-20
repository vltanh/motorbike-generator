import os
import imageio

path = 'output_128'
images = [  imageio.imread(f'{path}/{x}') 
            for i, x in enumerate(sorted(os.listdir(path)))
            if 'fake' in x
            and i % 5 == 0]
imageio.mimsave('output.gif', images, duration=1)
