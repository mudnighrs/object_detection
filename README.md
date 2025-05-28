
#   Dataset Structure

This project involves using the following dataset (stored in zip file) :
SSL dataset:-  https://drive.google.com/file/d/1MdmkhdkhNjXM_PZDaZ9kRuoaS80vsZ8_/view?usp=drivesdk

pre> <code>```text unzipped_dataset/ └── ssl_dataset/ ├── train.X1/ ├── train.X2/ ├── train.X3/ ├── train.X4/ ├── val.X/ └── Labels.json ```</code> </pre>


 `train.X*` directories contain unlabeled images for self-supervised training.
 `val.X` contains labeled validation data .

# 1. SimCLR-Based Self-Supervised Learning 

This project implements a simplified **SimCLR** pipeline using PyTorch for self-supervised representation learning on an image dataset. The pipeline includes dataset preparation, augmentations, contrastive training using the NT-Xent loss, and linear evaluation of learned representations.

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
with zipfile.ZipFile("/content/drive/MyDrive/ssl_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("unzipped_dataset")
2. SimCLR Transformations
Applies a combination of random cropping, flipping, color jitter, grayscale, blur, and tensor conversion.

3. Custom Dataset for Contrastive Learning
The SimCLRDataset returns two augmentations of each image using TwoCropTransform.

4. SimCLR Model
Uses a ResNet50 encoder and a projection head.

5. Contrastive Training
Trains the model using the NT-Xent loss. Training is done for a small number of epochs for demonstration.

6. Feature Extraction
Freezes the encoder and extracts features from the validation set.

7. Linear Evaluation
Trains a simple linear classifier on top of frozen features and evaluates it using accuracy and macro F1-score.


## Results 
Accuracy: 0.1428
Macro F1: 0.1132 



# 2. Self-Supervised Learning with Masked Autoencoders 

This project implements a Self-Supervised Learning (SSL) pipeline using a masked autoencoding approach based on a ResNet50 encoder and a custom decoder. It includes:

- Dataset preparation and loading
- Pretraining with masked image modeling
- Linear evaluation on labeled data
- Metric evaluation (accuracy, F1-score, confusion matrix)

## Preprocessing and Augmentation

Applies image augmentation (random crop, flip, color jitter, etc.) using torchvision.transforms.
Normalizes images using ImageNet mean and std.
Optionally masks a random 75% of image patches for SSL pretraining.
 
## Model Architecture

Encoder: ResNet50 (final FC layer removed)
Decoder: Transposed CNN layers to reconstruct 224x224 images
Linear Head: A simple linear classifier used for evaluation

## Pretraining (Masked Autoencoder)

Images are randomly masked and passed through the encoder. The decoder attempts to reconstruct the original image.

Loss: Mean Squared Error (MSE) between original and reconstructed images.

## Linear Evaluation

After pretraining, the encoder is frozen and a linear classifier is trained on the extracted features from the training dataset.
Loss: CrossEntropyLoss

## Results:
Accuracy: 0.0810,
F1-score: 0.0674

Model trained for 4 epochs, stopped at the 5th.
Plots training loss per epoch.
Prints evaluation metrics on validation set.
