# Mental Health Sentiment Analysis using Deep Learning

## 1. Introduction  
This project explores the use of deep learning models to classify text related to mental health into sentiment categories. It employs various neural network architectures including a multi-channel CNN, LSTM with L2 regularization, and BERT embeddings with XGBoost. The goal is to assess the models' performance and determine if there are significant differences in their predictive capabilities through hypothesis testing.

## 2. Dataset  
The dataset comprises 53,043 samples, with two primary columns:  
- **Text**: Sentences or phrases related to mental health.  
- **Labels**: Corresponding sentiment categories.

## 3. Preprocessing  
The data is preprocessed to remove null, missing values and tokenize input for neural network ingestion.
### Techniques Used:  
- **Stopword Removal**: Common stopwords (e.g., "and", "the") were removed using the NLTK library.  
- **Stemming**: Words were reduced to their root form using the Porter Stemmer.  
- **Tokenization and Padding**: Input text was tokenized and padded to a fixed length using a pre-trained tokenizer.  
- **BERT Embeddings**: Text was converted into high-dimensional vectors using BERT preprocessing and encoding layers from TensorFlow Hub.

## 4. Model Building  
The project employs three distinct deep learning models, each with unique architectural characteristics and strengths.

### 4.1 Multi-Channel CNN  
- **Architecture**:The multi-channel CNN is designed to capture various n-gram patterns in text by using multiple parallel convolutional layers.Three parallel Conv1D layers with varying kernel sizes to capture different n-gram features.
- **Input Layer**: Tokenized and padded input sequences.
- **Training Accuracy**: 99.23%  
- **Validation Accuracy**: 93.59%  
- **Test Accuracy**: 93.74%  

### 4.2 LSTM with L2 Regularization  
- **Architecture**:LSTM is a type of recurrent neural network (RNN) that excels at capturing sequential dependencies, making it ideal for text data where word order matters.Sequential LSTM layers with L2 regularization to mitigate overfitting.
- **Input Layer**: Sequential input for text sequences. 
- **Training Accuracy**: 99.26%  
- **Validation Accuracy**: 92.18%  
- **Test Accuracy**: 92.90%  

### 4.3 BERT with XGBoost  
- **Architecture**: BERT is a transformer-based model pre-trained on large text corpora, capable of generating contextual embeddings that capture rich semantic information. BERT embeddings feeding into an XGBoost classifier.
- **BERT Preprocessing Layer**: Tokenizes and formats input text for BERT.
- **BERT Encoder**: Generates deep embeddings (768 dimensions) for input text.
- **Training Accuracy**: 94.85%  
- **Validation Accuracy**: 83.43%  
- **Test Accuracy**: 83.92%  

## 5. Interactive Widgets  
An interactive interface was developed using IPyWidgets to allow users to input text and select a model for prediction.  
### Features:  
- **Text Input Box**: Accepts user input sentences.  
- **Model Selector Dropdown**: Allows switching between CNN, LSTM, and BERT models.  
- **Prediction Button**: Triggers sentiment prediction and displays the result.  
- **Dynamic Output**: Displays the predicted sentiment category.

## 6. Cross-Validation  
K-Fold Cross-Validation (k=5) was used to evaluate the generalizability of the models.  
### Results Summary:  
- **CNN Average Accuracy**: 93.58%  
- **LSTM Average Accuracy**: 92.90%  

## 7. Hypothesis Testing  
A paired t-test was performed based on the 5-fold cross validation data to determine if the CNN and LSTM models have statistically significant differences in performance.

### Results:  
- **T-Statistic**: -2.4131
- **P-Value**: 0.0733  
- **Significance Level (α)**: 0.05  
- If p-value < α: The null hypothesis is rejected, indicating a significant difference between CNN and LSTM.  
- If p-value ≥ α: Fail to reject the null hypothesis, indicating no significant difference.
Here we **"Fail to reject the null hypothesis: There is no significant difference between the models."**

## 8. Conclusion  
This project demonstrates the effectiveness of various deep learning approaches in sentiment analysis of mental health text. While the CNN achieved the highest accuracy, hypothesis testing validated whether this performance was statistically significant compared to the LSTM model. The user-friendly interface enhances accessibility for real-time text classification.
