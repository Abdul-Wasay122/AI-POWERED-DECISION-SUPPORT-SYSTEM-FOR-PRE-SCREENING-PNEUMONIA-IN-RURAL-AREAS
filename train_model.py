"""
Pneumonia Detection Model Training Script
=========================================
This script trains a deep learning model to detect pneumonia from chest X-ray images.
Uses Transfer Learning with ResNet50 pre-trained on ImageNet.

Author: FYP Abdul Awan
Dataset: Chest X-Ray Pneumonia (Balanced Split)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# ====================================
# CONFIGURATION
# ====================================
print("=" * 60)
print("      PNEUMONIA DETECTION MODEL TRAINING")
print("=" * 60)

# Paths
DATASET_PATH = './chest_xray'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'val')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

# Create output directories
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model Configuration
IMG_SIZE = (224, 224)  # ResNet50 standard input size
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

print(f"\n📂 Dataset Path: {DATASET_PATH}")
print(f"📊 Image Size: {IMG_SIZE}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")
print(f"📈 Learning Rate: {LEARNING_RATE}")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please make sure you have run the dataset balancing script first.")
    exit()

# ====================================
# DATA LOADING & AUGMENTATION
# ====================================
print("\n" + "=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

# Data Augmentation for training (helps prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1
    rotation_range=15,           # Randomly rotate images
    width_shift_range=0.1,       # Randomly shift images horizontally
    height_shift_range=0.1,      # Randomly shift images vertically
    shear_range=0.1,             # Shear transformation
    zoom_range=0.1,              # Random zoom
    horizontal_flip=True,        # Randomly flip images
    fill_mode='nearest'
)

# Validation and Test data (only rescaling, no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
print("\n📁 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification: NORMAL vs PNEUMONIA
    shuffle=True,
    seed=42
)

# Load validation data
print("📁 Loading validation data...")
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Load test data
print("📁 Loading test data...")
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n✅ Data loaded successfully!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Classes: {train_generator.class_indices}")

# ====================================
# MODEL CREATION
# ====================================
print("\n" + "=" * 60)
print("STEP 2: BUILDING MODEL")
print("=" * 60)

print("\n🔨 Creating ResNet50 model with transfer learning...")

# Load pre-trained ResNet50 (trained on ImageNet)
base_model = ResNet50(
    weights='imagenet',           # Use pre-trained weights
    include_top=False,            # Don't include the classification layer
    input_shape=(224, 224, 3)     # Input shape for RGB images
)

# Freeze base model layers (don't train them initially)
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)              # Pool features
x = Dense(512, activation='relu')(x)          # Fully connected layer
x = Dropout(0.5)(x)                           # Dropout for regularization
x = Dense(256, activation='relu')(x)          # Another FC layer
x = Dropout(0.3)(x)                           # More dropout
predictions = Dense(1, activation='sigmoid')(x)  # Binary output (0=NORMAL, 1=PNEUMONIA)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model created successfully!")
print(f"\n📊 Model Summary:")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ====================================
# TRAINING CALLBACKS
# ====================================
print("\n" + "=" * 60)
print("STEP 3: SETTING UP TRAINING")
print("=" * 60)

# Callback to save best model
checkpoint = ModelCheckpoint(
    'pneumonia_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Callback to stop if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Callback to reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

print("✅ Training callbacks configured:")
print("   - Model checkpointing (saves best model)")
print("   - Early stopping (stops if no improvement)")
print("   - Learning rate reduction (adapts learning rate)")

# ====================================
# TRAINING
# ====================================
print("\n" + "=" * 60)
print("STEP 4: TRAINING MODEL")
print("=" * 60)
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print("This may take 30-90 minutes depending on your hardware.\n")

start_time = datetime.now()

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

print(f"\n✅ Training completed!")
print(f"   Total time: {training_time//60:.0f} minutes {training_time%60:.0f} seconds")

# ====================================
# SAVE TRAINING HISTORY
# ====================================
print("\n" + "=" * 60)
print("STEP 5: SAVING TRAINING RESULTS")
print("=" * 60)

# Save training history to JSON
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}

with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
    json.dump(history_dict, f, indent=2)

print("✅ Training history saved")

# ====================================
# PLOT TRAINING HISTORY
# ====================================
print("\n📊 Creating training plots...")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_plots.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Training plots saved: training_plots.png")

# ====================================
# EVALUATION ON TEST SET
# ====================================
print("\n" + "=" * 60)
print("STEP 6: EVALUATING ON TEST SET")
print("=" * 60)

print("\n🔍 Evaluating model on test data...")

# Get predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels
true_classes = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

print(f"\n{'=' * 60}")
print("TEST SET RESULTS")
print(f"{'=' * 60}")
print(f"\nTotal Test Images: {len(true_classes)}")
print(f"Correct Predictions: {sum(true_classes == predicted_classes)}")
print(f"Incorrect Predictions: {sum(true_classes != predicted_classes)}")
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print(f"{'=' * 60}")

# ====================================
# CONFUSION MATRIX
# ====================================
print("\n📊 Creating confusion matrix...")

cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✅ Confusion matrix saved: confusion_matrix.png")

# ====================================
# CLASSIFICATION REPORT
# ====================================
print("\n📄 Generating classification report...")

# Get classification report
report = classification_report(true_classes, predicted_classes, 
                              target_names=CLASS_NAMES, digits=4)

print("\n" + report)

# Save report to file
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

print("✅ Classification report saved: classification_report.txt")

# ====================================
# PERFORMANCE SUMMARY
# ====================================
print("\n📄 Creating performance summary...")

summary = f"""======================================
    PNEUMONIA DETECTION RESULTS
======================================

DATASET INFORMATION:
- Training Images: {train_generator.samples:,}
- Validation Images: {val_generator.samples:,}
- Test Images: {test_generator.samples:,}

MODEL CONFIGURATION:
- Architecture: ResNet50 (Transfer Learning)
- Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}
- Epochs Trained: {len(history.history['accuracy'])}
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LEARNING_RATE}

FINAL TEST RESULTS:
- Overall Accuracy: {accuracy * 100:.2f}%
- Precision: {precision * 100:.2f}%
- Recall (Sensitivity): {recall * 100:.2f}%
- F1-Score: {f1 * 100:.2f}%

CONFUSION MATRIX:
              Predicted
          NORMAL  PNEUMONIA
Actual
NORMAL      {cm[0][0]:>4}      {cm[0][1]:>4}
PNEUMONIA   {cm[1][0]:>4}      {cm[1][1]:>4}

ERROR ANALYSIS:
- False Positives: {cm[0][1]} (said pneumonia, was normal)
- False Negatives: {cm[1][0]} (said normal, was pneumonia)
- True Positives: {cm[1][1]} (correctly identified pneumonia)
- True Negatives: {cm[0][0]} (correctly identified normal)

TRAINING TIME: {training_time//60:.0f} minutes {training_time%60:.0f} seconds

MODEL FILE: pneumonia_model.h5
======================================
"""

print(summary)

# Save summary
with open(os.path.join(RESULTS_DIR, 'performance_summary.txt'), 'w') as f:
    f.write(summary)

print("✅ Performance summary saved: performance_summary.txt")

# ====================================
# SAMPLE PREDICTIONS
# ====================================
print("\n📄 Creating sample predictions...")

# Get file names
test_filenames = test_generator.filenames

# Select 20 random samples
np.random.seed(42)
sample_indices = np.random.choice(len(true_classes), size=min(20, len(true_classes)), replace=False)

sample_predictions = "Sample Predictions (20 random test images):\n"
sample_predictions += "=" * 60 + "\n\n"

correct_count = 0
for i, idx in enumerate(sample_indices, 1):
    filename = test_filenames[idx]
    actual = CLASS_NAMES[true_classes[idx]]
    predicted = CLASS_NAMES[predicted_classes[idx]]
    confidence = predictions[idx][0] if predicted_classes[idx] == 1 else (1 - predictions[idx][0])
    is_correct = "✅ CORRECT" if actual == predicted else "❌ WRONG"
    
    if actual == predicted:
        correct_count += 1
    
    sample_predictions += f"{i}. {is_correct}\n"
    sample_predictions += f"   File: {filename}\n"
    sample_predictions += f"   Actual: {actual} | Predicted: {predicted}\n"
    sample_predictions += f"   Confidence: {confidence * 100:.1f}%\n\n"

sample_predictions += "=" * 60 + "\n"
sample_predictions += f"Accuracy in sample: {correct_count}/{len(sample_indices)} ({correct_count/len(sample_indices)*100:.1f}%)\n"

print(sample_predictions)

# Save sample predictions
with open(os.path.join(RESULTS_DIR, 'sample_predictions.txt'), 'w', encoding='utf-8') as f:
    f.write(sample_predictions)

print("✅ Sample predictions saved: sample_predictions.txt")

# ====================================
# FINAL SUMMARY
# ====================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

print(f"\n📦 Model saved: pneumonia_model.h5")
print(f"📁 Results saved in: {RESULTS_DIR}/")
print(f"\n📊 Generated files:")
print(f"   1. training_plots.png - Training history visualization")
print(f"   2. confusion_matrix.png - Performance breakdown")
print(f"   3. performance_summary.txt - Overall metrics")
print(f"   4. sample_predictions.txt - Example predictions")
print(f"   5. classification_report.txt - Detailed metrics")
print(f"   6. training_history.json - Raw training data")

print(f"\n🎯 Final Test Accuracy: {accuracy * 100:.2f}%")

if accuracy >= 0.90:
    print("🎉 Excellent performance! Your model is ready for deployment.")
elif accuracy >= 0.85:
    print("✅ Good performance! Model is acceptable for FYP.")
else:
    print("⚠️  Performance could be improved. Consider training longer or adjusting hyperparameters.")

print("\n📱 Next step: Run the web app!")
print("   Command: streamlit run app.py")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)