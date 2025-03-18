import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def print_classification_report(y_true, y_pred, classes):
    """Print classification report"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

def evaluate_model(model, test_gen):
    """Evaluate model on test set"""
    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    return results

def predict_and_visualize(model, test_gen, num_samples=5):
    """Make predictions and visualize results"""
    # Get a batch of images
    images, labels = next(test_gen)
    
    # Make predictions
    predictions = model.predict(images[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels[:num_samples], axis=1)
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    
    # Plot results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {class_names[true_classes[i]]}\nPred: {class_names[predicted_classes[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_model(model, model_path):
    """Save model to disk"""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """Load model from disk"""
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def plot_label_distribution(train_df, valid_df, test_df):
    """Etiket dağılımını görselleştirir"""
    try:
        print("Etiket dağılımı grafikleri oluşturuluyor...")
        
        # Eğitim seti dağılımı
        plt.figure(figsize=(12, 6))
        train_df['label'].value_counts().plot(kind='bar')
        plt.title('Eğitim Seti Etiket Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Görüntü Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/train_distribution.png')
        plt.close()
        print("Eğitim seti dağılımı grafiği kaydedildi")
        
        # Validasyon seti dağılımı
        plt.figure(figsize=(12, 6))
        valid_df['label'].value_counts().plot(kind='bar')
        plt.title('Validasyon Seti Etiket Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Görüntü Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/valid_distribution.png')
        plt.close()
        print("Validasyon seti dağılımı grafiği kaydedildi")
        
        # Test seti dağılımı
        plt.figure(figsize=(12, 6))
        test_df['label'].value_counts().plot(kind='bar')
        plt.title('Test Seti Etiket Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Görüntü Sayısı')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/test_distribution.png')
        plt.close()
        print("Test seti dağılımı grafiği kaydedildi")
        
        print("Tüm etiket dağılımı grafikleri başarıyla oluşturuldu!")
        
    except Exception as e:
        print(f"Etiket dağılımı grafikleri oluşturulurken hata oluştu: {str(e)}")
        print("Eğitim devam ediyor...") 