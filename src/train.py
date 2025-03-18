import os
import tensorflow as tf
from data_preprocessing import create_df, create_gens, show_images, plot_label_count
from model import create_model, unfreeze_model, get_model_summary
from callbacks import MyCallback
from utils import plot_training_history, evaluate_model, predict_and_visualize, save_model

# Model ve eğitim parametreleri
BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = 224

def main():
    try:
        print("Script started")
        print("Starting training process...")
        
        # Veri dizinini kontrol et
        data_dir = "Training Data"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")
            
        print(f"Data directory: {data_dir}")
        
        # Dataframe'leri oluştur
        print("Creating dataframes...")
        train_df, valid_df, test_df = create_df(
            os.path.join(data_dir, "train"),
            os.path.join(data_dir, "valid"),
            os.path.join(data_dir, "test")
        )
        print("Dataframes created successfully")
        
        # Etiket dağılımını görselleştir
        print("\nPlotting label distribution...")
        plot_label_count(train_df, "Training")
        plot_label_count(valid_df, "Validation")
        plot_label_count(test_df, "Test")
        
        # Model mimarisini oluştur
        print("\nCreating model architecture...")
        num_classes = len(train_df.label.unique())
        model = create_model(num_classes)
        get_model_summary(model)
        
        # Data generator'ları oluştur
        print("\nCreating data generators...")
        train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, BATCH_SIZE)
        print("Data generators created successfully")
        
        # Show sample images
        print("\nShowing sample images...")
        show_images(train_gen)
        
        # Callback'leri oluştur
        print("\nCreating callbacks...")
        callbacks = [
            MyCallback(
                model=model,
                patience=5,
                stop_patience=3,
                threshold=0.95,
                factor=0.5,
                batches=len(train_gen),
                epochs=EPOCHS,
                ask_epoch=10
            )
        ]
        print("Callbacks created successfully")
        
        # Modeli eğit
        print("\nStarting model training...")
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Eğitim sonuçlarını görselleştir
        print("\nPlotting training history...")
        plot_training_history(history)
        
        # Modeli değerlendir
        print("\nEvaluating model...")
        evaluate_model(model, test_gen)
        
        # Örnek tahminler yap
        print("\nMaking predictions on test images...")
        predict_and_visualize(model, test_gen, test_df)
        
        # Modeli kaydet
        print("\nSaving model...")
        save_model(model, "saved_models/animal_classifier.h5")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    print("Script finished") 