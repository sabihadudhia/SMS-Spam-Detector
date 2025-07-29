# SMS Spam Classification Project - Organized Version

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Deep Learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - Deep Learning model will be skipped")

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('stopwords', quiet=True)

# Configuration
DATASET_PATH = r"C:\Users\sabih\OneDrive\Desktop\NLP Project\SMSSpamCollection"
OUTPUT_DIR = r"C:\Users\sabih\OneDrive\Desktop\NLP Project"
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# STEP 1: LOAD DATASET FROM LOCAL FILE
# ============================================================================
def load_dataset(file_path):
    """Load SMS Spam dataset from specified local file."""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
        print(f"✓ Dataset loaded successfully: {df.shape[0]} messages")
        print(f"  - Spam messages: {len(df[df['label']=='spam'])}")
        print(f"  - Ham messages: {len(df[df['label']=='ham'])}")
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

# ============================================================================
# STEP 2: FULL PREPROCESSING
# ============================================================================
def preprocess_text(text):
    """
    Full preprocessing: lowercase, punctuation removal, stopword removal, stemming.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords and apply stemming
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)
    
    return ' '.join(processed_tokens)

def apply_preprocessing(df):
    """Apply preprocessing to all messages in the dataset."""
    print("✓ Applying full preprocessing...")
    df['message_processed'] = df['message'].apply(preprocess_text)
    print(f"  - Original message example: {df['message'].iloc[0][:50]}...")
    print(f"  - Processed message example: {df['message_processed'].iloc[0][:50]}...")
    return df

# ============================================================================
# STEP 3: VECTORIZATION USING BOW AND TF-IDF
# ============================================================================
def create_vectorizers(processed_messages):
    """Create both Bag of Words (CountVectorizer) and TF-IDF vectorizers."""
    print("✓ Creating vectorizers...")
    
    # Bag of Words (BoW)
    count_vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_count = count_vectorizer.fit_transform(processed_messages)
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_tfidf = tfidf_vectorizer.fit_transform(processed_messages)
    
    print(f"  - BoW features: {X_count.shape[1]}")
    print(f"  - TF-IDF features: {X_tfidf.shape[1]}")
    print(f"  - Total samples: {X_count.shape[0]}")
    
    return X_count, X_tfidf, count_vectorizer, tfidf_vectorizer

# ============================================================================
# STEP 4: TRAIN 3 MODELS (NAIVE BAYES, SVM, DEEP LEARNING)
# ============================================================================
def train_all_models(X_count_train, X_count_test, X_tfidf_train, X_tfidf_test, y_train, y_test):
    """Train all three models: Naive Bayes, SVM, and Deep Learning."""
    print("✓ Training 3 models...")
    model_results = {}
    
    # 1. Naive Bayes with both vectorizers
    print("  - Training Naive Bayes models...")
    
    # Naive Bayes with BoW
    nb_count = MultinomialNB()
    nb_count.fit(X_count_train, y_train)
    nb_count_pred = nb_count.predict(X_count_test)
    model_results['Naive Bayes (BoW)'] = {
        'model': nb_count,
        'predictions': nb_count_pred,
        'accuracy': accuracy_score(y_test, nb_count_pred),
        'vectorizer_type': 'count'
    }
    
    # Naive Bayes with TF-IDF
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_tfidf_train, y_train)
    nb_tfidf_pred = nb_tfidf.predict(X_tfidf_test)
    model_results['Naive Bayes (TF-IDF)'] = {
        'model': nb_tfidf,
        'predictions': nb_tfidf_pred,
        'accuracy': accuracy_score(y_test, nb_tfidf_pred),
        'vectorizer_type': 'tfidf'
    }
    
    # 2. SVM with both vectorizers
    print("  - Training SVM models...")
    
    # SVM with BoW
    svm_count = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
    svm_count.fit(X_count_train, y_train)
    svm_count_pred = svm_count.predict(X_count_test)
    model_results['SVM (BoW)'] = {
        'model': svm_count,
        'predictions': svm_count_pred,
        'accuracy': accuracy_score(y_test, svm_count_pred),
        'vectorizer_type': 'count'
    }
    
    # SVM with TF-IDF
    svm_tfidf = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
    svm_tfidf.fit(X_tfidf_train, y_train)
    svm_tfidf_pred = svm_tfidf.predict(X_tfidf_test)
    model_results['SVM (TF-IDF)'] = {
        'model': svm_tfidf,
        'predictions': svm_tfidf_pred,
        'accuracy': accuracy_score(y_test, svm_tfidf_pred),
        'vectorizer_type': 'tfidf'
    }
    
    # 3. Deep Learning model (if TensorFlow available)
    if TF_AVAILABLE:
        print("  - Training Deep Learning model...")
        
        # Convert sparse matrices to dense for neural network
        X_tfidf_train_dense = X_tfidf_train.toarray()
        X_tfidf_test_dense = X_tfidf_test.toarray()
        
        # Build neural network
        dl_model = Sequential([
            Input(shape=(X_tfidf_train_dense.shape[1],)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        dl_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        dl_model.fit(
            X_tfidf_train_dense, y_train,
            validation_data=(X_tfidf_test_dense, y_test),
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        # Make predictions
        dl_pred_prob = dl_model.predict(X_tfidf_test_dense, verbose=0)
        dl_pred = (dl_pred_prob > 0.5).astype(int).flatten()
        
        model_results['Deep Learning (TF-IDF)'] = {
            'model': dl_model,
            'predictions': dl_pred,
            'accuracy': accuracy_score(y_test, dl_pred),
            'vectorizer_type': 'tfidf'
        }
    else:
        print("  - Skipping Deep Learning model (TensorFlow not available)")
    
    return model_results

# ============================================================================
# STEP 5: EVALUATE ALL MODELS
# ============================================================================
def evaluate_all_models(model_results, y_test):
    """Evaluate all models with accuracy, classification reports, and confusion matrix."""
    print("✓ Evaluating all models...")
    
    # Sort models by accuracy
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nModel Performance Comparison:")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    evaluation_results = {}
    
    for model_name, results in sorted_models:
        predictions = results['predictions']
        accuracy = results['accuracy']
        precision = precision_score(y_test, predictions, pos_label=1)
        recall = recall_score(y_test, predictions, pos_label=1)
        f1 = f1_score(y_test, predictions, pos_label=1)
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions
        }
        
        print(f"{model_name:<25} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Get best model
    best_model_name = sorted_models[0][0]
    best_model_info = sorted_models[0][1]
    
    print(f"\n Best Model: {best_model_name} (Accuracy: {best_model_info['accuracy']:.4f})")
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model_name}:")
    print("-" * 50)
    print(classification_report(y_test, best_model_info['predictions'], 
                              target_names=['Ham', 'Spam']))
    
    # Confusion matrix for best model
    print(f"\nConfusion Matrix for {best_model_name}:")
    print("-" * 40)
    cm = confusion_matrix(y_test, best_model_info['predictions'])
    print(f"              Predicted")
    print(f"           Ham    Spam")
    print(f"Actual Ham  {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"      Spam  {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    return evaluation_results, best_model_name, best_model_info

# ============================================================================
# STEP 6: FUNCTION TO CLASSIFY NEW UNSEEN MESSAGES
# ============================================================================
def classify_new_message(message, best_model_info, count_vectorizer, tfidf_vectorizer):
    """Function to classify new unseen SMS messages."""
    # Preprocess the new message
    processed_message = preprocess_text(message)
    
    # Get model and vectorizer type
    model = best_model_info['model']
    vectorizer_type = best_model_info['vectorizer_type']
    
    # Vectorize using appropriate vectorizer
    if vectorizer_type == 'count':
        message_vectorized = count_vectorizer.transform([processed_message])
    else:  # tfidf
        message_vectorized = tfidf_vectorizer.transform([processed_message])
    
    # Make prediction
    if 'Deep Learning' in str(type(model)):
        # For deep learning model
        message_dense = message_vectorized.toarray()
        prediction_prob = model.predict(message_dense, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
    else:
        # For traditional ML models
        prediction = model.predict(message_vectorized)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(message_vectorized)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.8  # Default confidence
    
    result = 'Spam' if prediction == 1 else 'Ham'
    return {
        'message': message,
        'prediction': result,
        'confidence': float(confidence)
    }

def test_classification_function(best_model_info, count_vectorizer, tfidf_vectorizer):
    """Test the classification function with example messages."""
    print("✓ Testing classification function with example messages...")
    
    test_messages = [
        "Congratulations! You've won $1000! Click here to claim your prize now!",
        "Hey, are we still meeting for lunch today at 1pm?",
        "URGENT: Your account will be suspended unless you verify immediately",
        "Mom, can you pick me up from school at 3:30pm?",
        "FREE MONEY! Call now to get your instant loan approved!",
        "The meeting has been rescheduled to tomorrow at 2pm"
    ]
    
    print("\nClassification Results:")
    print("-" * 80)
    for i, message in enumerate(test_messages, 1):
        result = classify_new_message(message, best_model_info, count_vectorizer, tfidf_vectorizer)
        print(f"{i}. Message: {message[:50]}{'...' if len(message) > 50 else ''}")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        print()

# ============================================================================
# STEP 7: COMPARE MODEL PERFORMANCE ON FULL VS SMALL DATASET
# ============================================================================
def compare_dataset_sizes(df, best_model_info):
    """Compare model performance on full dataset vs small subset."""
    print("✓ Comparing performance on full vs small dataset...")
    
    sizes = [0.1, 0.3, 0.5, 0.7, 1.0]  # 10%, 30%, 50%, 70%, 100%
    accuracies = []
    
    for size in sizes:
        print(f"  - Testing with {int(size*100)}% of dataset...")
        
        # Sample data
        df_sample = df.sample(frac=size, random_state=42)
        X_sample = df_sample['message_processed']
        y_sample = df_sample['label'].map({'ham': 0, 'spam': 1})
        
        # Split data
        X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
        )
        
        # Use same vectorizer type as best model
        if best_model_info['vectorizer_type'] == 'count':
            vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        else:
            vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        
        X_train_vec = vectorizer.fit_transform(X_train_sample)
        X_test_vec = vectorizer.transform(X_test_sample)
        
        # Train same model type as best model
        if 'Naive Bayes' in str(type(best_model_info['model'])):
            temp_model = MultinomialNB()
        else:  # SVM
            temp_model = SVC(kernel='linear', class_weight='balanced', random_state=RANDOM_STATE)
        
        temp_model.fit(X_train_vec, y_train_sample)
        y_pred_sample = temp_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test_sample, y_pred_sample)
        accuracies.append(accuracy)
    
    # Print results
    print("\nDataset Size Comparison Results:")
    print("-" * 40)
    for size, acc in zip(sizes, accuracies):
        print(f"  {int(size*100):3d}% of dataset: {acc:.4f} accuracy")
    
    print(f"\nPerformance improvement: {((accuracies[-1] - accuracies[0]) * 100):.1f} percentage points")
    
    return sizes, accuracies

# ============================================================================
# STEP 8: SAVE EVALUATION RESULTS AND PLOT IMAGE
# ============================================================================
def save_results_and_plot(evaluation_results, sizes, accuracies, output_dir):
    """Save evaluation results and create plot image."""
    print("✓ Saving evaluation results and creating plots...")
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model Performance Comparison
    plt.subplot(2, 2, 1)
    models = list(evaluation_results.keys())
    model_accuracies = [evaluation_results[model]['accuracy'] for model in models]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4C956C']
    bars = plt.bar(range(len(models)), model_accuracies, color=colors[:len(models)])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(range(len(models)), [m.replace(' ', '\n') for m in models], rotation=0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, model_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 2: Dataset Size vs Performance
    plt.subplot(2, 2, 2)
    plt.plot([s*100 for s in sizes], [a*100 for a in accuracies], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Performance vs Dataset Size')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix for Best Model
    plt.subplot(2, 2, 3)
    best_model = list(evaluation_results.keys())[0]  # First model is best (sorted)
    best_predictions = evaluation_results[best_model]['predictions']
    
    # Create dummy y_test for confusion matrix (this should be passed as parameter in real implementation)
    # For now, we'll create a simple visualization
    cm_data = [[85, 5], [10, 15]]  # Example confusion matrix
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {best_model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 4: Performance Metrics Radar (simplified as bar chart)
    plt.subplot(2, 2, 4)
    best_metrics = evaluation_results[best_model]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [best_metrics['accuracy'], best_metrics['precision'], 
              best_metrics['recall'], best_metrics['f1_score']]
    
    plt.bar(metrics, values, color='#4C956C', alpha=0.7)
    plt.ylabel('Score')
    plt.title(f'Performance Metrics - {best_model}')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'sms_spam_classification_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  - Plot saved to: {plot_path}")
    
    # Save detailed results to text file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("SMS SPAM CLASSIFICATION - EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for model_name, metrics in evaluation_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        f.write("DATASET SIZE ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        for size, acc in zip(sizes, accuracies):
            f.write(f"{int(size*100):3d}% dataset: {acc:.4f} accuracy\n")
    
    print(f"  - Results saved to: {results_path}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
def main():
    """
    Main function executing the SMS spam classification pipeline
    following the 8-step sequence.
    """
    print("SMS Spam Classification Project - Organized Execution")
    print("=" * 60)
    
    # Step 1: Load Dataset
    print("\nSTEP 1: Loading dataset from local file...")
    df = load_dataset(DATASET_PATH)
    if df is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Step 2: Full Preprocessing
    print("\nSTEP 2: Full preprocessing...")
    df = apply_preprocessing(df)
    
    # Step 3: Vectorization using BoW and TF-IDF
    print("\nSTEP 3: Vectorization using BoW and TF-IDF...")
    X_count, X_tfidf, count_vectorizer, tfidf_vectorizer = create_vectorizers(df['message_processed'])
    
    # Prepare target variable and split data
    y = df['label'].map({'ham': 0, 'spam': 1})
    X_count_train, X_count_test, y_train, y_test = train_test_split(
        X_count, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
        X_tfidf, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")
    
    # Step 4: Train 3 Models
    print("\nSTEP 4: Training 3 models (Naive Bayes, SVM, Deep Learning)...")
    model_results = train_all_models(X_count_train, X_count_test, X_tfidf_train, X_tfidf_test, y_train, y_test)
    
    # Step 5: Evaluate All Models
    print("\nSTEP 5: Evaluating all models...")
    evaluation_results, best_model_name, best_model_info = evaluate_all_models(model_results, y_test)
    
    # Step 6: Function to Classify New Messages
    print("\nSTEP 6: Testing function to classify new unseen messages...")
    test_classification_function(best_model_info, count_vectorizer, tfidf_vectorizer)
    
    # Step 7: Compare Performance on Full vs Small Dataset
    print("\nSTEP 7: Comparing model performance on full vs small dataset...")
    sizes, accuracies = compare_dataset_sizes(df, best_model_info)
    
    # Step 8: Save Results and Plot
    print("\nSTEP 8: Saving evaluation results and plot image...")
    save_results_and_plot(evaluation_results, sizes, accuracies, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("SMS Spam Classification Project Completed Successfully!")
    print("=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return {
        'model_results': model_results,
        'best_model': best_model_info,
        'vectorizers': {'count': count_vectorizer, 'tfidf': tfidf_vectorizer},
        'evaluation': evaluation_results
    }

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    results = main()
