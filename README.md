# Sentiment-Explorer-Using-Simple-RNN
This project demonstrates how to use a Simple Recurrent Neural Network (RNN) to classify IMDb movie reviews as positive or negative.
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Steps to run in VS code](#steps)
## Introduction

Sentiment analysis is a common task in natural language processing (NLP) that involves determining the emotional tone behind a text. In this project, we use the IMDb dataset, a collection of movie reviews labeled as positive or negative, to train and evaluate a Simple RNN model.

---

## Dataset

The dataset is part of the TensorFlow/Keras library and can be directly loaded using:
python
``` bash
     from tensorflow.keras.datasets import imdb
```
---

## Model Architecture

The Simple RNN model consists of the following layers:

1. **Embedding Layer:** Converts word indices to dense vectors of fixed size.
2. **Simple RNN Layer:** Processes sequences of embeddings to capture temporal dependencies.
3. **Dense Layer:** Fully connected layer with a single neuron and sigmoid activation for binary classification.

## [Model Summary](#modelsummary) 
![rnnsummary](https://github.com/user-attachments/assets/f7e6eacb-f7eb-4ead-b846-db1a7c926193)


---

## Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib (optional, for visualization)

---
## Run the code using steps -

1. Clone the repository 
  ``` bash
   git clone <url_of_repository>
  ``` 
2. Select the kernel and run all the files.
3. Run main.py using commond
   ```bash
   streamlit run main.py
   ```
 4. Enter the movie review & classify  it as positive or negative.
##[Web app](#webapp)
-------------
 ![good review](https://github.com/user-attachments/assets/311a7e00-c423-4f76-af3b-a824b157b7b1)

  ![badreviw](https://github.com/user-attachments/assets/8b42e2e9-187c-4b67-8144-5ba5bf35d979)

   ---
