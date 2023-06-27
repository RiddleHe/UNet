# UNet

A U-Net model capable of segmenting any given image into 30 possible classes of objects. The core of the model, the convolutional blocks and the upsampling blocks that have residual connections from the former, is manually implemented.

## Technologies

- Implemented 4 layers of transposed convolutions to upsample the previous output so that the spatial features of the original 96x128 input is retained.

- Added residual connections from downsampling convolutional layers to the transposed convolutional layers to combine textural information with spatial information.

- Carefully maintained a U-shape of downsampling and upsampling layers so that the original dimensionality of previous layers were all successfully restored.

## Performance

- Trained for 5 epochs on a A100 GPU on 1060 images, decreasing the loss to 0.3909 and increase the accuracy to 0.8852.
