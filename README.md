# SMS Spam Detector

## Overview
Classify SMS messages as spam or ham (legitimate) using Natural Language Processing and machine learning. The project includes preprocessing, vectorization, model training, evaluation, and visualization.

## Features
- Text preprocessing: lowercase, stopword removal, stemming
- Vectorization: Bag of Words and TF-IDF
- Train multiple models: Naive Bayes, SVM, optionally Deep Learning
- Evaluate model performance with accuracy, precision, recall, F1-score
- Analyze effect of dataset size
- Real-time classification of new messages

## Technologies
- Python 3.8+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- NLTK
- TensorFlow (optional)

## Setup / Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/sms-spam-classification.git
cd sms-spam-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Set dataset path in the script if needed:
```bash
DATASET_PATH = r"your/path/to/SMSSpamCollection"
```

## Usage
Run the main script:
```bash
python sms_spam_classification.py
```

Outputs include evaluation results and visualizations
Classify new messages using the built-in function:
```bash 
result = classify_new_message("Congratulations! You've won a prize!", best_model_info, count_vectorizer, tfidf_vectorizer)
print(result)
```

## Project Structure
```bash
├── SMSSpamCollection                 # Dataset
├── sms_spam_classification.py        # Main script
├── evaluation_results.txt            # Model evaluation
├── sms_spam_classification_results.png # Visualization results
└── README.md                         # Documentation
```

## Example Output
- Accuracy, precision, recall, and F1-score for all models
- Confusion matrices and performance plots saved automatically
