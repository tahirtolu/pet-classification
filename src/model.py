import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

def create_model(num_classes, img_size=(224, 224)):
    """Create and compile the model"""
    print("\nCreating model architecture...")
    
    # Load pre-trained EfficientNetB0 model
    print("Loading pre-trained EfficientNetB0...")
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    print("Base model loaded successfully")
    
    # Freeze the base model layers
    base_model.trainable = False
    print("Base model layers frozen")
    
    # Create the model
    print("Building model architecture...")
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.016), 
              activity_regularizer=tf.keras.regularizers.l1(0.006),
              bias_regularizer=tf.keras.regularizers.l1(0.006), 
              activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(num_classes, activation='softmax')
    ])
    print("Model architecture built successfully")
    
    # Compile the model
    print("Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled successfully")
    
    return model

def unfreeze_model(model, unfreeze_layers=30):
    """Unfreeze some layers of the base model for fine-tuning"""
    print("\nUnfreezing layers for fine-tuning...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last 'unfreeze_layers' layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    print(f"Unfroze last {unfreeze_layers} layers")
    
    # Recompile the model with a lower learning rate
    print("Recompiling model with lower learning rate...")
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model recompiled successfully")
    
    return model

def get_model_summary(model):
    """Print model summary and return trainable parameters count"""
    print("\nModel Summary:")
    model.summary()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    return trainable_params 