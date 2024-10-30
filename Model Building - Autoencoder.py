import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('C:/Users/TRIVESH SHARMA/Desktop/python/Autoencoder/archive/creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)  # Features (all columns except 'Class')
y = df['Class']  # Target (Class column)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Filter only genuine (normal) data for training the autoencoder
X_train_genuine = X_train[y_train == 0]  # Train on only genuine (Class == 0)

# Autoencoder model
CODE_DIM = 2
INPUT_SHAPE = X_train_genuine.shape[1]

input_layer = Input(shape=(INPUT_SHAPE,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(16, activation='relu')(x)
code = Dense(CODE_DIM, activation='relu')(x)
x = Dense(16, activation='relu')(code)
x = Dense(64, activation='relu')(x)
output_layer = Dense(INPUT_SHAPE, activation='relu')(x)

autoencoder = Model(input_layer, output_layer, name='anomaly')

# Compile the model
autoencoder.compile(loss='mae', optimizer=Adam())

# Model callbacks
model_name = "anomaly.weights.h5"
checkpoint = ModelCheckpoint(model_name,
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1,
                              restore_best_weights=True)

callbacks = [checkpoint, earlystopping]

# Train the model with callbacks
history = autoencoder.fit(X_train_genuine, X_train_genuine,
                          epochs=25, batch_size=64,
                          validation_data=(X_test, X_test),
                          shuffle=True,
                          callbacks=callbacks)

# Model summary
autoencoder.summary()

# --- Performance Evaluation Code ---

# Plotting Training and Validation Loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss values by Epoch")
plt.show()

# Predicting Reconstructions and Calculating Reconstruction Errors
reconstructions = autoencoder.predict(X_test, verbose=0)
reconstruction_error = np.mean(np.abs(reconstructions - X_test), axis=1)  # Using MAE

# Creating DataFrame with Reconstruction Errors
recons_df = pd.DataFrame({
    'error': reconstruction_error,
    'y_true': y_test
}).reset_index(drop=True)
# Creating the y_pred column based on a threshold value
threshold = 0.2  # Set your desired threshold here
recons_df['y_pred'] = recons_df['error'] > threshold

# Displaying the first few rows
pd.set_option('display.max_columns', None)  # To display all columns
print(recons_df.head())

# Threshold Tuning Function
def thresholdTuning(df, iterations):
    thresh_df = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': []
    }
    
    for i in range(iterations):
        thresh_value = df['error'].quantile(i/iterations)
        preds = df['error'] > thresh_value
        cr = classification_report(df['y_true'], preds, output_dict=True)
        acc = cr['accuracy']
        prec = cr['macro avg']['precision']
        rc = cr['macro avg']['recall']
        
        thresh_df['threshold'].append(thresh_value)
        thresh_df['accuracy'].append(acc)
        thresh_df['precision'].append(prec)
        thresh_df['recall'].append(rc)
        
        print(f"Threshold: {thresh_value:.4f}\tAccuracy: {acc:.3f}\t\tPrecision: {prec:.3f}\tRecall Score: {rc:.3f}")
        
    return pd.DataFrame(thresh_df)

# Tuning Threshold
thresh_df = thresholdTuning(recons_df, 10)
threshold = thresh_df[thresh_df['recall'] == thresh_df['recall'].max()]['threshold'].values[0]
print(f"Threshold with Maximum Recall: {threshold:.6f}")

# Plotting Accuracy, Precision, and Recall by Threshold Values
plt.figure(figsize=(10,8))
plt.plot(thresh_df['threshold'], thresh_df['accuracy'], label='accuracy')
plt.plot(thresh_df['threshold'], thresh_df['precision'], label='precision')
plt.plot(thresh_df['threshold'], thresh_df['recall'], label='recall')
plt.axvline(x=threshold, color='r', linestyle='dashed')
plt.xlabel('Threshold')
plt.ylabel('Metrics')
plt.title('Metrics by Threshold Values')
plt.legend()
plt.show()

# Plotting Reconstruction Error for Random Sample
temp = recons_df.sample(frac=0.01, random_state=42).reset_index(drop=True)
plt.figure(figsize=(8,6))
sns.scatterplot(data=temp, x=temp.index, y='error', hue='y_true')
plt.axhline(y=threshold, color='r', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Error by Sample')
plt.legend()
plt.show()

# Assigning Predictions Based on Error Threshold
recons_df['y_pred'] = recons_df['error'] > threshold

# Evaluation Metrics
print(classification_report(recons_df['y_true'], recons_df['y_pred']))

# Confusion Matrix
cm = confusion_matrix(recons_df['y_true'], recons_df['y_pred'])
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt='.6g')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Additional Metrics (Recall and Accuracy)
print(f"Recall Score: {recall_score(recons_df['y_true'], recons_df['y_pred'])*100:.3f}%")
print(f"Accuracy Score: {accuracy_score(recons_df['y_true'], recons_df['y_pred'])*100:.3f}%")
