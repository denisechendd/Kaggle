# Contradictory, My Dear Watson
The Kaggle competition [link](https://www.kaggle.com/c/contradictory-my-dear-watson)

## Competition Goal
The task is to conduct the classification prediction to match the given class of contradiction, entailment, or neutral from the given hypothesis. The output is evaluated by the accuracy metric, namely the matched percentage with each class.
## Dataset
There are fifteen different languages appearing in the premise-hypothesis dataset, such as Arabic, Bulgarian, Chinese, German, Greek, English, Spanish, French, Hindi, Russian, Swahili, Thai, Turkish, Urdu, and Vietnamese.
In the Dataset, given the premise **He came, he opened the door and I remember looking back and seeing the expression on his face, and I could tell that he was disappointed.**, there would be different hypothesis matched with the premise. <br>
**Hypothesis 1:** <br>
- Just by the look on his face when he came through the door I just knew that he was let down.<br>
The hypothesis is truely matched with the premise. Then, the sentence is seen as entailment.
**Hypothesis 2:** <br>
- He was trying not to make us feel guilty but we knew we had caused him trouble.<br>
The hypothesis is somewhat true, but we can confirm the information given the premise. Therefore, this hypothesis is considered neutral.

**Hypothesis 3:** <br>
- He was so excited and bursting with joy that he practically knocked the door off it's frame.<br>
The hypothesis is not true, and extremely opposite with the premise. Then, the hypothesis is considered contradiction.

**File descriptions**
  - train.csv - the training set. This file contains the ID, premise, hypothesis, and label, as well as the language of the text and its two-letter abbreviation
  - test.csv - the test set. This file contains the ID, premise, hypothesis, language, and language abbreviation, without labels.
## Achievement
The final accuracy metric is 0.68. I ranked top 44% out of 89 competitors.
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
  The `mask_type_probs` parameter is tweaked through the parameter tuning. The `mask_type_probs` parameter is set with the tuple (a, b, c). a: the proportion to be replaced by the mask token. b: the proportion to be kept as it it. c: the proportion to be replaced by a random token in the tokenizer's vocabulary. The loss function is calculated by the sparse categorical cross entropy.
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

#### multivariate_LSTM
  - Data Preprocessing
    - fill the (shop, item) tuple set with 0 during training period when there are no sales_train
    - fill the (shop, item) tuple set with 0 during validatoin period when there are no sales_train
  - Features
    - item average price, shop_item_sum on monthly and yearly basis, items sold count on monthly basis, total items sold sold on monthly basis,
    - **time lag features:**
      - sum up shop total items sold every month on previous 1, 2, 3 ,and 6 months
      - sum up item total sold count every month on previous 1, 2, 3 ,and 6 months
  - ttl training feature shape: (1175113, 22)
  - input feature shape (1019418, 1, 21)
    - (# num of samples, # num of time steps, # num of features)

  - modelling  <br>
    <img src="image/LSTM1.png" width="80%" height="80%"> <br>
    <img src="image/LSTM2.png" width="80%" height="80%"> <br>

    LSTM solves **Vanishing Gradient Problem**, which shows that gradient shrink exponentially and have small values near 0. LSTM both have a gating mechanism to regulate the flow of information like remembering the context over multiple time steps. There are **Input gate**, **Forget Gate**, and **Output Gate**
    - Input gate: update the cell state. When it's 0, it would not update the cell state. When it's 1, it would update the cell, meaning data is important.
    - Forget Gate: Keep or drop information from the previous hidden state. When it's open (1), it keeps the previous dataset. On the other hand, when it's close, it updates the new data and drop the past data.
    - Output Gate: The output gate decides what the next hidden state should be. Hidden state contains information on previous inputs and is used for prediction.

    First, we pass the previous hidden state and the current input into a sigmoid function. Then we pass the newly modified cell state to the tanh function. We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. The output is the hidden state.

  - Final output
    - Training mse: 0.0237
    - Training mae: 0.0562
    - Validation mse: 0.0410
    - Validation mae: 0.0281
