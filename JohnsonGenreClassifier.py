#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import ast
import plotly.express as px
import numpy as np
import numpy
import os
import librosa
import random
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from tensorflow import keras


# In[2]:


# Functions that will help us extract our audio files and make them unique vectors
def get_mfcc(wav_file_path):
  y, sr = librosa.load(wav_file_path, offset=0, duration=30)
  mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
  return mfcc

def get_melspectrogram(wav_file_path):
  y, sr = librosa.load(wav_file_path, offset=0, duration=30)
  melspectrogram = numpy.array(librosa.feature.melspectrogram(y=y, sr=sr))
  return melspectrogram

def get_chroma(wav_file_path):
  y, sr = librosa.load(wav_file_path)
  chroma = numpy.array(librosa.feature.chroma_stft(y=y, sr=sr))
  return chroma

def get_tonnetz(wav_file_path):
  y, sr = librosa.load(wav_file_path)
  tonnetz = numpy.array(librosa.feature.tonnetz(y=y, sr=sr))
  return tonnetz
#
# Combines the functions above to make the final vector
def get_feature(file_path):
  # Extracting MFCC feature
  mfcc = get_mfcc(file_path)
  mfcc_mean = mfcc.mean(axis=1)
  mfcc_min = mfcc.min(axis=1)
  mfcc_max = mfcc.max(axis=1)
  mfcc_feature = numpy.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = get_melspectrogram(file_path)
  melspectrogram_mean = melspectrogram.mean(axis=1)
  melspectrogram_min = melspectrogram.min(axis=1)
  melspectrogram_max = melspectrogram.max(axis=1)
  melspectrogram_feature = numpy.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = get_chroma(file_path)
  chroma_mean = chroma.mean(axis=1)
  chroma_min = chroma.min(axis=1)
  chroma_max = chroma.max(axis=1)
  chroma_feature = numpy.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = get_tonnetz(file_path)
  tntz_mean = tntz.mean(axis=1)
  tntz_min = tntz.min(axis=1)
  tntz_max = tntz.max(axis=1)
  tntz_feature = numpy.concatenate( (tntz_mean, tntz_min, tntz_max) ) 
  
  feature = numpy.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
  return feature


# In[3]:


def calc_performance_multi_label(y_vals_true, y_vals_pred, threshold=0.5):
    """
    Calculate multi-label performance metrics with handling for undefined metrics and thresholding predictions.
    """
    # Threshold predictions (convert probabilities to binary labels)
    y_vals_pred_binary = (y_vals_pred > threshold).astype(int)

    # Hamming Loss
    hamming = hamming_loss(y_vals_true, y_vals_pred_binary)

    # Precision, Recall, F1-Score with zero_division=0 to handle undefined metrics
    precision_micro = precision_score(y_vals_true, y_vals_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_vals_true, y_vals_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(y_vals_true, y_vals_pred_binary, average='micro', zero_division=0)

    precision_macro = precision_score(y_vals_true, y_vals_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_vals_true, y_vals_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(y_vals_true, y_vals_pred_binary, average='macro', zero_division=0)

    # Classification report (optional)
    report = classification_report(y_vals_true, y_vals_pred_binary, zero_division=0, output_dict=True)

    # Results dictionary
    results = {
        'hamming_loss': hamming,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'classification_report': report
    }

    return results


# In[4]:


# Import the data 

# Directories for both train and val
train = '/users/PAS2038/ejohnson/osc_classes/PHYSICS_5680_OSU/materials/Final_Project/train_files'
val = '/users/PAS2038/ejohnson/osc_classes/PHYSICS_5680_OSU/materials/Final_Project/val_files'

# Note - this is just the information for the audio files
df_train_raw = pd.read_csv('/users/PAS2038/ejohnson/osc_classes/PHYSICS_5680_OSU/materials/Final_Project/train.tsv', sep='\t')
df_val_raw = pd.read_csv('/users/PAS2038/ejohnson/osc_classes/PHYSICS_5680_OSU/materials/Final_Project/valid.tsv', sep='\t')

# Convert 'genres' from strings to lists
df_train_raw['genres'] = df_train_raw['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_val_raw['genres'] = df_val_raw['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Clean out rows where 'genres' is empty
df_train = df_train_raw[df_train_raw['genres'].apply(lambda x: len(x) > 0)]
df_val = df_val_raw[df_val_raw['genres'].apply(lambda x: len(x) > 0)]

# Explode the 'genres' column to separate each genre into its own row
df_train_exploded = df_train.explode('genres')
df_val_exploded = df_val.explode('genres')

# Get all unique genres
train_genres= df_train_exploded['genres'].str.strip().dropna().unique().tolist()
val_genres= df_val_exploded['genres'].str.strip().dropna().unique().tolist()
print(len(train_genres))
print("Unique genres (train) :", train_genres)
print("Unique genres (val) :", val_genres)

print("Train :")
print(df_train.head(10))
print("")
print("Val :")
print(df_val.head(10))


# In[5]:


# Combine all unique genres from both train and val datasets
all_genres = sorted(list(set([genre for genres in df_train['genres'] for genre in genres] + 
                             [genre for genres in df_val['genres'] for genre in genres])))

# Create a mapping from genre to index
genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}

# Function to generate the binary array for each row
def genres_to_array(genres, mapping):
    array = [0] * len(mapping)  # Initialize array of 0s with length equal to the number of unique genres
    for genre in genres:
        array[mapping[genre]] = 1  # Set the index corresponding to the genre to 1
    return array

# Apply the function to both train and val datasets
df_train['genre_array'] = df_train['genres'].apply(lambda x: genres_to_array(x, genre_to_index))
df_val['genre_array'] = df_val['genres'].apply(lambda x: genres_to_array(x, genre_to_index))

# Example: Print the resulting DataFrames
print("Train DataFrame with Genre Arrays:")
print(df_train[['genres', 'genre_array']].head())
print()
print("Validation DataFrame with Genre Arrays:")
print(df_val[['genres', 'genre_array']].head())


# In[6]:


# Grouping the arrays together
df_train['genre_array_tuple'] = df_train['genre_array'].apply(tuple)

grouped = df_train.groupby('genre_array_tuple')['id'].apply(list).reset_index()
grouped.columns = ['genre_array', 'IDs']

print(grouped)


# In[7]:


# Count the occurrences of each genre for each set of data
genre_counts_train = df_train_exploded['genres'].value_counts().reset_index()
genre_counts_train.columns = ['genre', 'Count']
print("For training set :")
print(genre_counts_train.head(10))

fig = px.bar(genre_counts_train , x='genre', y='Count', title="Genre Distribution for Training Set",
             labels={'genre': 'Music Genre', 'Count': 'Number of Occurrences'},
             color='Count')
fig.write_image("genre_distribution_train.png")

genre_counts_val = df_val_exploded['genres'].value_counts().reset_index()
genre_counts_val.columns = ['genre', 'Count']
print("For validation set : ")
print(genre_counts_val.head(10))

fig = px.bar(genre_counts_val , x='genre', y='Count', title="Genre Distribution for Val Set",
             labels={'genre': 'Music Genre', 'Count': 'Number of Occurrences'},
             color='Count')
fig.write_image("genre_distribution_val.png")


# In[8]:


top_genres = genre_counts_train.sort_values(by='Count', ascending=False)['genre'].head(10).tolist()
print(top_genres)


# In[9]:


all_genres = sorted(list(set([genre for genres in df_train['genres'] for genre in genres] + 
                             [genre for genres in df_val['genres'] for genre in genres])))

# Create a mapping from genre to index
genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}

# Function to generate the binary array for each genre
def genres_to_array(genres, mapping):
    array = [0] * len(mapping)  # Initialize array of 0s with length equal to the number of unique genres
    for genre in genres:
        if genre in mapping:  # Only include genres in the mapping
            array[mapping[genre]] = 1  # Set the index corresponding to the genre to 1
    return array


# Create a dictionary to store the file IDs per genre
genre_file_map = defaultdict(list)


# In[10]:


# Loop through the dataframe and check if the genre belongs to the top_genres list
for idx, row in df_train.iterrows():
    genres = row['genres']
    file_id = row['id']
    
    # Check if all genres of this file are in the top_genres list
    if all(genre in top_genres for genre in genres):  # Ensure only tracks with top genres are considered
        for genre in genres:
            genre_file_map[genre].append(file_id)

selected_files = defaultdict(list)

for genre, files in genre_file_map.items():
    selected_files_for_genre = random.sample(files, min(1000, len(files)))  # Limit to 1000 files
    selected_files[genre] = selected_files_for_genre


# In[11]:


# Now apply the one-hot encoding to the selected files and store them
df_train_filtered = df_train[df_train['id'].isin([file_id for file_list in selected_files.values() for file_id in file_list])]

# One-hot encode the genres for the selected training files
df_train_filtered['genre_array'] = df_train_filtered['genres'].apply(lambda x: genres_to_array(x, genre_to_index))

# Grouping the arrays together for the final dataset
df_train_filtered['genre_array_tuple'] = df_train_filtered['genre_array'].apply(tuple)

# Group by genre arrays to count the occurrences of each
grouped = df_train_filtered.groupby('genre_array_tuple')['id'].apply(list).reset_index()
grouped.columns = ['genre_array', 'IDs']

# Print the final result and genre counts
print(grouped)

# Count the occurrences of each genre in the filtered training set
genre_counts_train = df_train_filtered['genres'].explode().value_counts().reset_index()
genre_counts_train.columns = ['Genre', 'Count']

print("For filtered training set:")
print(genre_counts_train)


# In[12]:



top_genres = genre_counts_train['Genre'].head(10).tolist()
print("Top 10 Genres:", top_genres)

# Create a mapping from genre to index for the top 10 genres only
genre_to_index = {genre: idx for idx, genre in enumerate(top_genres)}

# Function to generate the binary array for each genre (one-hot encoded for top 10 genres)
def genres_to_array(genres, mapping):
    array = [0] * len(mapping)  # Initialize array of 0s with length equal to the number of top genres
    for genre in genres:
        if genre in mapping:  # Only include genres that are in the top genres list
            array[mapping[genre]] = 1  # Set the index corresponding to the genre to 1
    return array


#count_t = 0
#n = 50  # Number of files to process

# Initialize empty lists to store the data for the dictionary
ids = []
genre_arrays = []
audio_arrays = []

# Loop through the files and extract the required data
print("Loading Training Files")
for file in os.listdir(train):
#    if count_t >= n: 
#        break
    
    if file.endswith('.opus'):
        file_id = int(file.split('.')[0])  # Extract file ID from the filename

        # Get the file path
        file_path = os.path.join(train, file)

        # Extract features using the get_feature function
        audio_features = get_feature(file_path)

        # Get the genre array for this file (ensure only top 10 genres)
        genres = df_train[df_train['id'] == file_id]['genres'].values[0]
        genre_array = genres_to_array(genres, genre_to_index)

        # Store the results in the lists
        ids.append(file_id)
        genre_arrays.append(genre_array)
        audio_arrays.append(audio_features)

        print("Loading file :",file)
#        count_t += 1

# Create the dictionary with the collected data
data_dict = {
    'id': ids,
    'genre_array': genre_arrays,  # One-hot encoded genres (only top 10)
    'audio_array': audio_arrays
}

# Print the dictionary to verify the data
print(data_dict)

# Convert to NumPy arrays for training
X_train = np.array(audio_arrays)
y_train = np.array(genre_arrays)


# In[13]:


# Initialize variables for the validation dataset
#count_v = 0
#n = 50  # Limit the number of files processed
val_arrays = []

# Initialize empty lists to store the data for the dictionary
ids_val = []
genre_arrays_val = []
audio_arrays_val = []

print("Loading Validation Files")
for file in os.listdir(val):
#    if count_v >= n:  # Limit the number of files processed
#        break
    
    if file.endswith('.opus'):
        file_id = int(file.split('.')[0])  # Extract file ID from the filename

        # Get the file path
        file_path = os.path.join(val, file)

        # Get the genres for this file from the df_val DataFrame
        genres = df_val[df_val['id'] == file_id]['genres'].values[0]

        # Only process files where all genres are in top_genres
        if all(genre in top_genres for genre in genres):  # Check if all genres are in top_genres
            # Extract features using the get_feature function
            audio_features_val = get_feature(file_path)

            # Generate the one-hot encoded genre array for this file
            genre_array_val = genres_to_array(genres, genre_to_index)

            # Store the results in the lists
            ids_val.append(file_id)
            genre_arrays_val.append(genre_array_val)
            audio_arrays_val.append(audio_features_val)

            print("Loading file :", file)
#            count_v += 1

# Create the dictionary with the collected data
data_dict_val = {
    'id': ids_val,
    'genre_array': genre_arrays_val,  # One-hot encoded genres (only top 10)
    'audio_array': audio_arrays_val
}

# Print the dictionary to verify the data
print("First few entries in the dictionary:", data_dict_val)

# Convert to NumPy arrays for training
X_val = np.array(audio_arrays)
y_val = np.array(genre_arrays)


# In[19]:


num_genres = len(top_genres)

# Split validation data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, shuffle=True)

# Input layer
inputs = keras.Input(shape=(X_train.shape[1],), name="feature")

# Hidden layers
x = keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dropout(0.5)(x)  # Drop 50% of neurons
x = keras.layers.Dense(128, activation="relu", name="dense_2")(x)
x = keras.layers.Dense(64, activation="relu", name="dense_3")(x)

# Output layer (adjusted to the number of genres)
outputs = keras.layers.Dense(num_genres, activation="sigmoid", name="predictions")(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(),  # For multi-label classification
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    epochs=64,
    verbose=1
)

test_accuracy, test_loss, test_auc = model.evaluate(X_test,y_test)
print("Test accuracy, loss, auc :",test_accuracy, test_loss, test_auc)


# In[22]:


df_history = pd.DataFrame(history.history)
print(df_history)

# ACCURACY
fig = px.line(df_history, y=['accuracy','val_accuracy'], title='Accuracy vs Epoch')
fig.write_image("accuracy_vs_epoch.png")
#
# Loss
fig = px.line(df_history, y=['loss','val_loss'], title='Loss vs Epoch')
fig.write_image("loss_vs_epoch.png")


# In[34]:


# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance using the updated function
results = calc_performance_multi_label(y_vals_true=y_test, y_vals_pred=y_pred)
print(results)

# Print metrics
print("Hamming Loss :", results['hamming_loss'])
print("Micro Precision :", results['precision_micro'], "Recall :", results['recall_micro'], "F1 :",results['f1_micro'])
print("Macro Precision :", results['precision_macro'], "Recall :", results['recall_macro'], "F1 :",results['f1_macro'])


# In[ ]:




