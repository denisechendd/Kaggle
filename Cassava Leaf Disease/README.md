# Cassava Leaf Disease Classification
The Kaggle competition [link](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)

## Competition Goal
Cassava is the main food crop grown in Africa to overcome the bad weather. However, the crops are suffered from the disease,  and have a poor yield. The task is to classify the cassava images into 5 categories including Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), and Healthy. The evaluation metric is based on the categorization accuracy.
## Dataset
The data contains 21,367 labeled images as the train dataset from the survey collection in Uganda, and the test set contains 15,000 images. The train and test images are stored in tfrecord format.

**File descriptions**
  - train.csv - the training set. This file contains the image_id, and the label.
  - test.csv - the test set. This file contains the image_id.
## Achievement
The final accuracy metric is 0.903 on test dataset. I ranked top 5% out of 3947 competitors, and achieved the silver medal.
## Approach

### Model Training Pipeline
  - Image Preprocessing
    - crop, transpose, flip, and rotate images
    - Load data into batches
  - Dataset:
    - Train data: 80% original dataset
    - Validation data: 20% original dataset

  - Model (**tf_efficientnet_b4_ns, resnext50_32x4d, VisionTransformer**)
    - Train the model in the 2-fold training dataset, and calculate the training loss and validation accuracy
    - Loss is calculated in CrossEntropyLoss.
    - Apply label smoothing as a regularization approach to minimize the gap of the largest logit with the rest
    - Save the trained model over each epoch

#### resnext50_32x4d model training performance over 1 fold
    Epochs | Train Loss | Valid Loss | Valid Accuracy
    --- | --- | --- | ---
    1 | 0.488 | 0.426 | 0.858
    2 | 0.482 | 0.417 | 0.864
    3 | 0.421 | 0.4 | 0.874
    4 | 0.43 | 0.361 | 0.884
    5 | 0.362 | 0.359 | 0.883
    6 | 0.37 | 0.357 | 0.887
    7 | 0.332 | 0.355 | 0.89
    8 | 0.333 | 0.346 | 0.889
    9 | 0.297 | 0.347 | 0.889
    10 | 0.3 | 0.344 | 0.892


#### tf_efficientnet_b4_ns model training performance over 1 fold
    Epochs | Train Loss | Valid Loss | Valid Accuracy
    --- | --- | --- | ---
    1 | 0.436 | 0.364 | 0.877
    2 | 0.414 | 0.338 | 0.882
    3 | 0.389 | 0.339 | 0.883
    4 | 0.359 | 0.323 | 0.888
    5 | 0.32 | 0.33 | 0.885
    6 | 0.314 | 0.349 | 0.879
    7 | 0.284 | 0.337 | 0.886
    8 | 0.288 | 0.334 | 0.886
    9 | 0.269 | 0.334 | 0.889
    10 | 0.284 | 0.339 | 0.886

### Inference
#### Methods
- Perform 2 fold validation on train dataset.
- Ensemble three pretrained-models with VisionTransformer, tf_efficientnet_b4_ns, and resnext50_32x4d.
- Apply weights to the respective models' predictions, and TTA of 3 to the predicted data in batches. **Test Time Augmentation (TTA) is the improved method for model predictions on test dataset. Instead of predicting the output from the actual image, TTA is the method to predict the augmented images several times, and computes the average of the probability from each image output.**
- Compute the mean of the predicted value from each fold.

Methods | Models | Loss function | Loss Value | TTA | Accuracy on Validation set | Accuracy on Test set
--- | --- | --- | --- | --- | --- | ---
1 | 4 * efficient net | Log loss | 0.23 | 4 | 0.93 | 0.898
2 | 6 * efficient net | KLDivLoss | 0.54 | 6 | 0.94 | 0.898
3 | 4 * efficient net + 1*VisionTransformer | Log loss | 0.273 | 4 | 0.922 | 0.903
4 | 4 * efficient net + 2*resnext50_32x4d | Log loss | 0.285 | 4 | 0.934 | 0.895
5 | 4 * efficient net + 2 * resnext50_32x4d + 1*VisionTransformer | Log loss | 0.255 | 4 | 0.927 | 0.903

## Challenge
- The test accuracy is not improved from 0.898 after changing the loss function or concatenating more trained models from different epochs
- The model is required with high computation resource such as GPU or TPU and it takes around 4.5 hours to train the model over 10 epochs on 2 fold train data.

## feedback
- Model Ensembling from different deep learning models would improve the test accuracy from 0.898 to 0.903
- Apply Test Time Augmentation (TTA) to model training on augmented dataset would regularize the model performance
