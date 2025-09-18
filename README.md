# Music Genre Classifier

This project is a **multi-label music genre classification model** built with Python and Keras.  
It was developed for my Physics 5680 course final project at The Ohio State University.  

The model uses the **Jamendo dataset** and extracts four unique audio features to predict the genre of a track.  
It achieves an accuracy of **82%**, loss of **1.40**, and **AUC of 0.86** on the 10 most populated genres.  

---

## Project Structure

```
music-genre-classifier/
│
├── src/
│   └── genre_classifier.py
│
├── reports/
│   ├── Big_Data_Report.pdf
│   └── Poster.pdf
│
├── requirements.txt
└── README.md 
```
---

## Motivation

Streaming services rely on algorithms to recommend new music.  
This project demonstrates how machine learning can be applied to classify genres,  
forming a foundation for recommendation systems.  

As a musician, I wanted to explore audio data in a way that was both fun and technically challenging.  

---

## Dataset

The project uses the [**Jamendo dataset**](https://mtg.github.io/jamendo-dataset/),  
which contains **55,000+ tracks** with metadata across **genre, instrument, and mood** categories.  

- Only the **10 most populated genres** were used.  
- Dataset was split into **72% training, 18% testing, 10% validation**.  
- Around **500 tracks** were used due to time constraints, though the dataset supports ~8,000. 

---

##  Features Extracted

Each audio file is processed into a feature vector of length 498 using:  

- **MFCC** (Mel-Frequency Cepstral Coefficients)  
- **Mel Spectrogram**  
- **Chroma Vector**  
- **Tonnetz (Tonal Centroid)**  

These features are concatenated and used as input to the classifier.  

---

## Model

A **Fully Convolutional Network (FCN)** was implemented using Keras:  

- Input: 498-length feature vector  
- Hidden layers: Dense (256, 128, 64) with ReLU activations  
- Dropout (50%) to reduce overfitting  
- Output: Sigmoid activation for multi-label classification  

**Training setup:**  
- Optimizer: Adam (`1e-4` learning rate)  
- Loss: Binary Crossentropy  
- Epochs: 64 (with early stopping)  
- Metrics: Accuracy, AUC  
