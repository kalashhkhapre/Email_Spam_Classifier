# Email Spam Classifier

## Overview
This project implements an Email Spam Classifier using various machine learning and deep learning techniques. The classifier processes raw email text, converts it into numerical features using TF-IDF vectorization, and applies different algorithms to classify emails as spam or ham (not spam). The implemented models include:
* Naïve Bayes (MultinomialNB)
* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest
* LSTM (Long Short-Term Memory) neural network

## Features
* Text preprocessing including cleaning, tokenization, and normalization
* Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency)
* Multiple classification algorithms implemented
* Performance comparison of different models
* Easy-to-use pipeline for training and prediction
* Model persistence using pickle files
* Email content analysis and prediction

## Results
The different models achieved the following accuracy scores on the test set:

| Model | Accuracy |
|-------|----------|
| Naïve Bayes | 90.34% |
| SVM | 97.54% |
| Logistic Regression | 97.10% |
| Random Forest | 96.84% |
| LSTM | 99.39% |

## Requirements
* Python 3.x
* scikit-learn
* TensorFlow/Keras (for LSTM)
* pandas
* numpy
* nltk
* matplotlib (for visualization)
* seaborn

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models
Run the Jupyter notebook `Email_Spam_Classifier.ipynb` to train all models and save them as pickle files.

### Making Predictions
Use the saved models to predict whether an email is spam or ham:
```python
import pickle

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
with open('SVM_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load email content
with open('email.txt', 'r') as file:
    email_content = file.read()

# Vectorize the email
email_vectorized = tfidf_vectorizer.transform([email_content])

# Make prediction
prediction = svm_model.predict(email_vectorized)
print("Spam" if prediction[0] == 1 else "Not Spam")
```

## File Structure
```
email-spam-classifier/
├── dataset/                    # Dataset directory
│   └── emails.csv              # Email dataset
├── models/                     # Saved models
│   ├── NB_model.pkl            # Naïve Bayes model
│   ├── SVM_model.pkl           # SVM model
│   ├── LOGISTIC_model.pkl      # Logistic Regression model
│   ├── RANDOM_FOREST_model.pkl # Random Forest model
│   ├── lstm_model.pkl          # LSTM model
│   └── vectorizer.pkl          # TF-IDF vectorizer
├── test/                       # Test emails
│   ├── ham/                    # Non-spam test emails
│   └── spam/                   # Spam test emails
├── Email_Spam_Classifier.ipynb # Main Jupyter notebook
└── README.md                   # Project documentation
```

## Data Preprocessing
The email data goes through several preprocessing steps:
1. Removal of punctuation
2. Tokenization
3. Stopword removal
4. TF-IDF vectorization

## Model Training
The notebook includes:
1. Data exploration and visualization
2. Text preprocessing pipeline
3. Model training and evaluation
4. Performance comparison
5. Model persistence

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
MIT License
