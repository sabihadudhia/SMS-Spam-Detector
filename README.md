# SMS Spam Detector

## Overview
The purpose of this task is the development of an NLP system that can collect a labelled dataset of SMS text messages classified as either spam or ham (legitimate), preprocess the data, train the preprocessed data on a machine learnig model using a supervised model (such as Naive Bayes or Support Vector Machines), test the model and use it to classify new text messages as spam or ham confidently. 
The chosen dataset: UC Irvie Machine Learning Repository with 5574 labelled data points.

## Results
The NLP-based system to classify SMS messages as spam or legitimate, achieved strong overall performance across multiple models. SVM with TF-IDF was the best approach, delivering the highest accuracy (~98%) and most balanced precision–recall performance, making it the most reliable classifier. Naïve Bayes also performed well as a fast and efficient baseline, while the deep learning model showed slightly lower results due to limited data but has potential for improvement with larger datasets. The model performance generally improved with increasing dataset size, stabilizing around 50–70%, though a drop at full data suggests possible noise or imbalance. Feature engineering played a crucial role, with TF-IDF providing better term importance and n-grams improving spam phrase detection. Overall, classical machine learning models proved highly effective and computationally efficient for this task, demonstrating strong robustness on noisy, real-world SMS data.

SMS Spam Classification Visualizations

<img width="4470" height="2965" alt="sms_spam_classification_results" src="https://github.com/user-attachments/assets/e70ccc24-2c71-48da-bc24-4bfba07d4b51" />


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
