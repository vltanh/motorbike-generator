# Usage

## Dataset

Because we are using the ImageDataLoader, it is required that the folder looks like this:
```
  data/
    train_data/
      folder_A/
        002.png
        abc.png
        ...
```

i.e. just make sure that the images are in a **subfolder** (folder_A in this example) of a **root folder** (train_data) (when train we will point the path to the root folder).

## Train

To train:

```
  python main.py --dataset folder --dataroot data/train_data --cuda --niter 100 --outf output_128 --imageSize 128 --ngf 128 --ndf 32 --nz 100
```

where:
+ We will be using ```dataset``` of type ```folder``` (images taken from folders), with root path ```dataroot``` (in this case ```data/train_data``` as said above)
+ It will be train on GPU with ```cuda``` flag.
+ Number of iterations is 100
+ The weights and intermediate visualization will be save in ```outf``` (in this case, ```output_128```).
+ Generates image of size 128
+ Number of (initial) channels (this is a hyperparam) in the Generator will be 128, Discriminator will be 32. It is suggested that ngf = 4 * ndf for better stabilization (tested this, other value would fk up).
+ Dimension of the input random vector is 100.
