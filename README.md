
#   Dataset Structure

This project  involves using the following  dataset  (stored in zip file) :
SSL dataset:-  https://drive.google.com/file/d/1MdmkhdkhNjXM_PZDaZ9kRuoaS80vsZ8_/view?usp=drivesdk
After unzipping, the expected directory structure is:

```
unzipped_dataset/
└── ssl_dataset/
 └── train.X1/
 └── train.X2/
 └── train.X3/
 └── train.X4/
 └── val.X/
 └── Labels.json
```

 `train.X*`  directories contain unlabeled images for self-supervised training
 `val.X`  contains labeled validation data 

# 1. SimCLR-Based Self-Supervised Learning 

This project implements a simplified SimCLR pipeline using PyTorch for self-supervised representation learning on an image dataset The pipeline includes dataset preparation, augmentations , contrastive training using the NT-Xent loss, and linear evaluation of learned representations

---

##  Features

-  Unzips and loads image data using `torchvision.datasets.ImageFolder`
-  Prepares SimCLR-style augmentations
-  Implements the SimCLR model with ResNet50 backbone
-  Trains with NT-Xent (contrastive) loss
-  Extracts embeddings and evaluates using a linear classifier
-  Reports accuracy and macro-F1 score

---

## Training Pipeline Overview

1. Unzipping the Dataset

2. SimCLR Transformations
Applies a combination of random cropping, flipping, color jitter , grayscale , blur and tensor conversion

3. Custom Dataset for Contrastive Learning
The SimCLRDataset returns two augmentations of each image using TwoCropTransform

4. SimCLR Model
Uses a ResNet50 encoder and a projection head.

5. Contrastive Training
Trains the model using the NT-Xent loss  Training is done for a small number of epochs for demonstration 

6. Feature Extraction
Freezes the encoder and extracts features from the validation set

7. Linear Evaluation
Trains a simple linear classifier on top of frozen features and evaluates it using accuracy and macro F1-score


## Results 
Accuracy: 0.1428
Macro F1: 0.1132 



# 2. Self-Supervised Learning with Masked Autoencoders 

This project implements  a Self-Supervised Learning (SSL) pipeline using a masked  autoencoding approach based on a ResNet50 encoder and a  custom decoder
It includes:

- Dataset preparation and loading
- Pretraining with masked image modeling
- Linear evaluation on labeled data
- Metric evaluation (accuracy, F1-score)

## Preprocessing and Augmentation

Applies image augmentation (random crop, flip, color jitter, etc.) using torchvision.transforms.
Normalizes  images using  ImageNet mean and std.
Optionally  masks a random 75% of image  patches for SSL pretraining.
 
## Model Architecture

Encoder: ResNet50 (final FC layer removed)
Decoder: Transposed CNN layers  to reconstruct 224x224 images
Linear Head: A  simple  linear classifier  used for evaluation

## Pretraining (Masked Autoencoder)

Images are randomly masked and passed through the encoder. The decoder attempts to reconstruct the original image.

Masking Strategy: Randomly  masks 75%  of the image patches ( nonlearnable masking).

Forward Pass:
Masked image → Encoder → Decoder → Reconstructed image
Loss:
Mean Squared Error  between original  and reconstructed image

Optimization:
Adam optimizer  on encoder and decoder weights

Note: Pretraining was run  for 4 epochs and  then stopped

## Linear Evaluation

After pretraining, the encoder is frozen and a linear classifier is trained on the extracted features from the training dataset
Loss: CrossEntropyLoss

Frozen Encoder: Encoder weights are frozen
Linear Head: A simple linear layer  added on top of the encoder for classification
Loss: Cross-entropy loss
Optimizer: Adam (for linear head only)
Training: Linear classifier trained on labeled data for 5 epochs.
Tracking: Training loss logged and plotted per epoch

## Results:
Accuracy: 0.0810,
F1-score: 0.0674

