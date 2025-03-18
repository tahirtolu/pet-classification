import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, patience=5, stop_patience=3, threshold=0.95, factor=0.5, batches=0, epochs=0, ask_epoch=10):
        super(MyCallback, self).__init__()
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.batches = batches
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch
        self.stop_training = False
        self.best_epoch = 0
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.ask_epoch = self.ask_epoch_initial
        self.epoch_times = []
        self.epoch_start_time = time.time()
        print("\nEğitim başlıyor...")
        print(f"Toplam epoch sayısı: {self.epochs}")
        print(f"Her {self.ask_epoch} epoch'ta bir devam etmek isteyip istemediğiniz sorulacak")
        print("-" * 50)

    def on_train_end(self, logs=None):
        print("\n" + "="*50)
        print("Eğitim Tamamlandı!")
        print(f"Toplam Epoch Sayısı: {len(self.epoch_times)}")
        print(f"Ortalama Epoch Süresi: {np.mean(self.epoch_times):.2f} saniye")
        print(f"En İyi Model Epoch {self.best_epoch+1}'de Kaydedildi")
        print(f"En İyi Accuracy: {self.best_accuracy:.4f}")
        print(f"En İyi Loss: {self.best_loss:.4f}")
        print("="*50 + "\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.pbar = tqdm(total=self.batches, desc=f'Epoch {epoch+1}/{self.epochs}', 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def on_batch_end(self, batch, logs=None):
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'accuracy': f"{logs.get('accuracy', 0):.4f}"
            })

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
            
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Epoch sonuçlarını göster
        print("\n" + "="*50)
        print(f"Epoch {epoch+1}/{self.epochs} Sonuçları:")
        print(f"Epoch Süresi: {epoch_time:.2f} saniye")
        print(f"Eğitim Loss: {logs.get('loss', 0):.4f}")
        print(f"Eğitim Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f"Validasyon Loss: {logs.get('val_loss', 0):.4f}")
        print(f"Validasyon Accuracy: {logs.get('val_accuracy', 0):.4f}")
        print("="*50 + "\n")
        
        # En iyi modeli kaydet
        if logs.get('val_accuracy', 0) > self.best_accuracy:
            self.best_accuracy = logs.get('val_accuracy', 0)
            self.best_loss = logs.get('val_loss', 0)
            self.best_epoch = epoch
            self.model.save('saved_models/best_model.h5')
            print(f"En iyi model kaydedildi! (Epoch {epoch+1})")
            print(f"En iyi accuracy: {self.best_accuracy:.4f}")
            print(f"En iyi loss: {self.best_loss:.4f}")
        
        # Eğitimi durdurma kontrolü
        if epoch + 1 >= self.ask_epoch and not self.stop_training:
            self.stop_training = True
            print("\nEğitimi durdurmak ister misiniz? (y/n)")
            response = input()
            if response.lower() == 'y':
                self.model.stop_training = True
                print("\nEğitim durduruldu!")
                print(f"En iyi model Epoch {self.best_epoch+1}'de kaydedildi")
                print(f"En iyi accuracy: {self.best_accuracy:.4f}")
                print(f"En iyi loss: {self.best_loss:.4f}")
            else:
                self.stop_training = False
                self.ask_epoch += self.ask_epoch_initial
                print("\nEğitim devam ediyor...") 