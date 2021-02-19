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

### modelling
#### Transformer XLMRoberta
  - Data Preprocessing (Data Batches)
    - tokenize the tuple input of the premise and hypothesis
    - Pad the tokens with length of 80
    - xlm-roberta token masking: tokens with higher counts will be masked based on the probability
    - words would be masked with 3 conditions regarding keeping the original words, being masked, and random masked words replacement
  - Dataset:
    - Train data: 70% original dataset
    - Validation data: 30% original dataset
    - Test data

  - Model (xlm Roberta base)
    - Predict the tokens to be masked, and calculate the loss of the masked tokens
    - Predict the labels from train, test, validation set. Run through the fine tune process after 3 epochs.
  - Parameter Tuning Approach <br>
  The `mask_type_probs` parameter is tweaked through the parameter tuning. The `mask_type_probs` parameter is set with the tuple (a, b, c).<br>
  a: the proportion to be replaced by the mask token. <br>
  b: the proportion to be kept as it is. <br>
  c: the proportion to be replaced by a random token in the tokenizer's vocabulary. <br>
  **The loss function is calculated by the sparse categorical cross entropy.** <br>
  In the mask_type_probs=(0.8, 0.1, 0.1), the train and validation loss is around 1.1. On the other hand, regarding the mask_type_probs=(0.3, 0.6, 0.1),
  the train loss is around 1.1, and the validation loss is around 1.05. The model with the mask_type_probs=(0.8, 0.1, 0.1) is overfitting. When the proportion
  to be replaced by the mask token is higher, the model performance is much better.
    - mask_type_probs=(0.8, 0.1, 0.1)

    Epochs | Train Loss | Valid Loss
    --- | --- | ---
    1 | 1.098 | 1.09
    2 | 1.13 | 1.084
    3 | 1.107 | 1.091
    - mask_type_probs=(0.3, 0.6, 0.1)

    Epochs | Train Loss | Valid Loss
    --- | --- | ---
    1 | 1.115 | 1.072
    2 | 1.104 | 1.041
    3 | 1.113 | 1.06

### Inference
#### Methods
- Perform 2 fold validation on train dataset.
- Ensemble three models with VisionTransformer, tf_efficientnet_b4_ns, and resnext50_32x4d.
- Apply weights to the respective models' predictions, and TTA of 3 to the predicted data in batches. **Test Time Augmentation (TTA) is the improved method for model predictions on test dataset. Instead of predicting the output from the actual image, TTA is to take the augmented images several times, and compute the average of the probability from the output of each image.**
- Compute the mean of the predicted value from each fold.

Methods | Models | Loss function | Loss Value | TTA | Accuracy on Validation set | Accuracy on Test set
--- | --- | --- | --- | --- | --- | ---
1 | 4 * efficient net | Log loss | 0.23 | 4 | 0.93 | 0.898
2 | 6 * efficient net | KLDivLoss | 0.54 | 6 | 0.94 | 0.898
3 | 4 * efficient net + 1*VisionTransformer | Log loss | 0.273 | 4 | 0.922 | 0.903
4 | 4 * efficient net + 2*resnext50_32x4d | Log loss | 0.285 | 4 | 0.934 | 0.895
5 | 4 * efficient net + 2 * resnext50_32x4d + 1*VisionTransformer | Log loss | 0.255 | 4 | 0.927 | 0.903

## Challenge
- It takes around 20 minutes to run through each epoch with GPU. The fine tune process of xlm-roberta-base model requires high computation resource.
