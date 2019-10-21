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

## Train

To train:

```
  python main.py --dataset folder --dataroot data/train_data --cuda --niter 100 --outf output_128 --imageSize 128 --ngf 128 --ndf 32 --nz 100
```
