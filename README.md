# U-Net_on_OxfordIIITPet

This is small project, for study U-Net-style architecture

## Dataset

For this project I used [Oxford pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset has 7930 images of dogs and cats, with mask for segmentation training.

## Model

Model is simply U-Net-style network, with encoder based on convolution-relu-batchnorm block, and max pooling, and decoder with deconvolution-convolution and concatenation step.

## Result

I trained this network on Kaggle kernel with Tesla P100 GPU. After 15 epochs, results are pretty neat.

![Abyssian](https://github.com/KordianChi/U-Net_on_OxfordIIITPet/blob/main/results/example_1_org.png) ![Abyssian](https://github.com/KordianChi/U-Net_on_OxfordIIITPet/blob/main/results/example_1_pred.png) ![Abyssian](https://github.com/KordianChi/U-Net_on_OxfordIIITPet/blob/main/results/example_1_target.png)
