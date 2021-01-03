# Contradictory, My Dear Watson
The Kaggle competition [link](https://www.kaggle.com/c/contradictory-my-dear-watson)

## Competition Goal
The task is to conduct the classification prediction to match the given class of contradiction, entailment, or neutral from the given hypothesis. The output is evaluated by the accuracy metric, namely the matched percentage with each class.
## Dataset
There are fifteen different languages appearing in the premise-hypothesis dataset, such as Arabic, Bulgarian, Chinese, German, Greek, English, Spanish, French, Hindi, Russian, Swahili, Thai, Turkish, Urdu, and Vietnamese.
In the Dataset, given the premise **He came, he opened the door and I remember looking back and seeing the expression on his face, and I could tell that he was disappointed.**, there would be different hypothesis matched with the premise. <br>
**Hypothesis 1:** <br>
- Just by the look on his face when he came through the door I just knew that he was let down. <br>
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
  The `mask_type_probs` parameter is tweaked through the parameter tuning. The `mask_type_probs` parameter is set with the tuple (a, b, c).<br>
  a: the proportion to be replaced by the mask token. <br>
  b: the proportion to be kept as it it. <br>
  c: the proportion to be replaced by a random token in the tokenizer's vocabulary. <br>
  **The loss function is calculated by the sparse categorical cross entropy.**
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
