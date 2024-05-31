# README

## Overview

This project aims to build and evaluate a machine learning model for classifying customer complaints and opinions based on their intent and emotion. The model predicts the priority level of each complaint, helping businesses manage and respond to customer feedback more effectively.

## Data

The dataset used for this project consists of customer complaints and opinions, including columns for intent and emotion. The dataset is cleaned and preprocessed to remove missing values and normalize text.

## Preprocessing

1. **Text Cleaning**:
   - Convert text to lowercase.
   - Remove special characters and numbers.
   - Tokenize text and remove stopwords.

2. **Tokenization and Padding**:
   - Tokenize the text data using TensorFlow's `Tokenizer`.
   - Convert text sequences into padded sequences for uniform input length.

3. **Label Encoding**:
   - Encode the target labels (intent) using `LabelEncoder` for compatibility with the model.

## Model Architecture

### LSTM Model

1. **Embedding Layer**:
   - Converts the input tokens into dense vectors of fixed size.

2. **LSTM Layer**:
   - Long Short-Term Memory (LSTM) layer to capture the temporal dependencies in the text data.

3. **Dense Layer**:
   - Fully connected layer with a softmax activation function to output class probabilities.

### RoBERTa Model

1. **Pretrained Tokenizer**:
   - Uses the `RobertaTokenizer` from Hugging Face's Transformers library to tokenize the text.

2. **RoBERTa Sequence Classification Model**:
   - Utilizes the `RobertaForSequenceClassification` model for fine-tuning on the classification task.

3. **Optimizer and Scheduler**:
   - Optimizer: AdamW with a learning rate scheduler to adjust the learning rate during training.

## Training and Evaluation

1. **Training**:
   - The model is trained using the training dataset, with validation performed at each epoch to monitor performance and prevent overfitting.

2. **Early Stopping**:
   - Implements early stopping to halt training when the validation loss stops improving, preventing overfitting.

3. **Evaluation**:
   - The model is evaluated on a test dataset to measure its accuracy and overall performance.

## Priority Mapping

A custom mapping function is applied to determine the priority of each complaint based on its intent and emotion. The priorities are categorized as `very high`, `high`, `mild`, `low`, or `unknown`.

## Dependencies

- Python
- pandas
- scikit-learn
- TensorFlow
- PyTorch
- transformers (Hugging Face)
- nltk
- tqdm

## Usage

1. **Preprocess Data**:
   - Load and preprocess the dataset using the provided functions.

2. **Train Model**:
   - Train the LSTM or RoBERTa model using the training script.

3. **Evaluate Model**:
   - Evaluate the model on the test dataset to determine its accuracy and performance.

4. **Priority Mapping**:
   - Apply the priority mapping function to categorize the priority of each complaint.

## Results

The model's performance is measured in terms of accuracy and a detailed classification report is generated. The results help in understanding the model's ability to correctly classify the intents and emotions of customer complaints and opinions.

## Conclusion

This project demonstrates the use of machine learning techniques to classify customer feedback and prioritize it based on intent and emotion. The implementation of LSTM and RoBERTa models provides a comprehensive approach to text classification in customer service applications.
