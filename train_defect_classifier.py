"""
Binary Image Classification for Casting Product Defect Detection
Using Pretrained ResNet50 with Transfer Learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CastingDefectClassifier:
    """
    Binary classifier for detecting defects in casting products
    Uses transfer learning with ResNet50
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def build_model(self, learning_rate=0.0001):
        """
        Build model using pretrained ResNet50 with custom classification head
        """
        # Load pretrained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build custom classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layer
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocess for ResNet50
        x = keras.applications.resnet50.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def prepare_data(self, data_dir):
        """
        Prepare train and validation data generators
        Expected structure:
        data_dir/
            train/
                ok_front/
                def_front/
            test/
                ok_front/
                def_front/
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training data
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Validation data
        val_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Test data
        test_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def train(self, train_gen, val_gen, epochs=30):
        """
        Train the model with callbacks
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, train_gen, val_gen, epochs=10):
        """
        Fine-tune the base model by unfreezing some layers
        """
        # Unfreeze the base model
        base_model = self.model.layers[4]  # ResNet50 base
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        # Continue training
        history_fine = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return history_fine
    
    def evaluate(self, test_gen):
        """
        Evaluate model on test set
        """
        # Get predictions
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        # Classification report
        class_names = list(test_gen.class_indices.keys())
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return y_true, y_pred, predictions, cm
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, predictions, save_path='roc_curve.png'):
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
        plt.close()
    
    def visualize_predictions(self, test_gen, num_samples=16, 
                            save_path='predictions_sample.png'):
        """
        Visualize sample predictions
        """
        test_gen.reset()
        batch_images, batch_labels = next(test_gen)
        predictions = self.model.predict(batch_images, verbose=0)
        
        class_names = list(test_gen.class_indices.keys())
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(batch_images))):
            axes[i].imshow(batch_images[i])
            
            true_label = class_names[int(batch_labels[i])]
            pred_label = class_names[int(predictions[i] > 0.5)]
            confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
                            color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction samples saved to: {save_path}")
        plt.close()
    
    def save_model(self, filepath='defect_classifier_model.h5'):
        """
        Save the trained model
        """
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("CASTING PRODUCT DEFECT DETECTION SYSTEM")
    print("="*60)
    
    # Configuration
    DATA_DIR = './casting_data'  # Update this path to your dataset location
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 30
    FINE_TUNE_EPOCHS = 10
    
    # Initialize classifier
    classifier = CastingDefectClassifier(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠️  Data directory not found: {DATA_DIR}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
        print("\nExtract it and update the DATA_DIR variable in the script.")
        return
    
    print("\n1. Preparing data...")
    train_gen, val_gen, test_gen = classifier.prepare_data(DATA_DIR)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Class indices: {train_gen.class_indices}")
    
    print("\n2. Building model...")
    classifier.build_model()
    print("\nModel architecture:")
    classifier.model.summary()
    
    print("\n3. Training model (Transfer Learning)...")
    classifier.train(train_gen, val_gen, epochs=INITIAL_EPOCHS)
    
    print("\n4. Fine-tuning model...")
    classifier.fine_tune(train_gen, val_gen, epochs=FINE_TUNE_EPOCHS)
    
    print("\n5. Evaluating on test set...")
    y_true, y_pred, predictions, cm = classifier.evaluate(test_gen)
    
    print("\n6. Generating visualizations...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(cm, list(test_gen.class_indices.keys()))
    classifier.plot_roc_curve(y_true, predictions)
    classifier.visualize_predictions(test_gen)
    
    print("\n7. Saving model...")
    classifier.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - best_model.h5 (best model during training)")
    print("  - defect_classifier_model.h5 (final model)")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - predictions_sample.png")


if __name__ == "__main__":
    main()
