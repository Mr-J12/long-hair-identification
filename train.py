import pandas as pd
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import argparse
import os
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 15 
DATA_CSV = os.path.join('data', 'labels.csv')
MODELS_DIR = 'models'
OUTPUT_DIR = 'output'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_model(task):
    """Builds a model with fine-tuning enabled."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    if task == 'age':
        output_layer = Dense(1, activation='linear', name='age_output')(x)
        loss = 'mse'
        metrics = ['mae']
    else:
        output_layer = Dense(1, activation='sigmoid', name=f'{task}_output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
        
    model = Model(inputs=base_model.input, outputs=output_layer)

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss=loss, 
        metrics=metrics
    )
    return model

def plot_history(history, task):
    """Plots training and validation history."""
    metric = 'accuracy' if task in ['gender', 'hair'] else 'mae'
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(f'{task.capitalize()} Model {metric.capitalize()}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{task.capitalize()} Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{task}_training_history_v2.png'))
    print(f"Saved new training plot to {os.path.join(OUTPUT_DIR, f'{task}_training_history_v2.png')}")

def train(task):
    """Main training function."""
    print(f"--- Starting training for task: {task} ---")
    df = pd.read_csv(DATA_CSV)
    
    y_col = {'age': 'age', 'gender': 'gender', 'hair': 'hair_length'}[task]
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filepath', y_col=y_col,
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='raw'
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='filepath', y_col=y_col,
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='raw'
    )
    
    model = build_model(task)
    model.summary()
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[lr_scheduler]
    )

    model_path = os.path.join(MODELS_DIR, f'{task}_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    plot_history(history, task)
    print(f"--- Finished training for task: {task} ---\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for age, gender, or hair length prediction.")
    parser.add_argument('--task', type=str, required=True, choices=['age', 'gender', 'hair'],
                        help="The task to train the model for.")
    args = parser.parse_args()
    
    train(args.task)