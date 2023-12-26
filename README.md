
This repository contains 4 deep learning implementations, split in two folders. Below are the descriptions of the tasks and approaches used in these implementations.

## File 1, Task 1: Character Recognition using Siamese Networks

### Overview
- **Dataset**: Omniglot dataset with images of handwritten characters.
- **Goal**: Recognize characters by comparing a query image to a set of candidate images, measuring the top-1 and top-3 performance of the model.

### Approach
- **Model**: Siamese Network.
- **Technique**: Meta-learning using few-shot learning methods for character recognition.
- **Implementation Details**: 
    - **Triplet Loss**: Enhances model's ability to map similar images close and dissimilar images far apart.
    - **Balanced Batch Sampler**: Used for efficient training with minibatches containing N classes and M samples per class.
    - **Cosine Similarity**: Employed for high dimensional space compatibility and accurate similarity measurement between images.

## File 1, Task 2: Activity Recognition from Accelerometer Data

### Overview
- **Dataset**: Data from 30 participants engaged in 6 different activities, captured through accelerometers.
- **Goal**: Predict the activity of a person based on sensor data from accelerometers and gyroscopes.

### Approach
- **Model**: Recurrent Neural Network with a stack of two bidirectional LSTM layers followed by two fully connected layers.
- **Technique**: Time-series classification to account for data across timestamps.
- **Implementation Details**: 
    - **LSTM Layers**: Allows retaining longer memory crucial for sequential data.
    - **Mini-Batches**: Processes data in sequences of 1000 time steps with 9 features.
    - **Bidirectional Learning**: Enhances learning by processing data in both forward and backward directions.
    - **Loss Calculation**: Utilizes categorical cross-entropy for accurate loss measurement.

## File 2, Task 1: Graph-based Article Retrieval

### Overview
- **Dataset**: Wiki-CS, a graph representation of Wikipedia articles in Computer Science.
- **Goal**: Efficiently retrieve articles using a graph neural network that accounts for the complex relations between articles.

### Dataset Properties
- Contains 10701 nodes (articles) and 251927 edges (hyperlinks).
- Each node features a 300-dimensional embedding based on important words in the article.
- The average outdegree of nodes is 23.54, with 522 nodes labeled for training and 5348 for testing.

### Approach
- **Model**: Attention-based Graph Neural Network (GNN)
- **Key Features**: 
    - Utilizes the attention mechanism to focus on neighbors sharing similar context or topic.
    - Employs a unique way of aggregating neighbor embeddings using learned attention coefficients.

## File 2, Task 2: VAE for Anomaly Detection in Article Database

### Overview
- **Dataset**: Extended Wiki-CS dataset with additional anomalous articles.
- **Goal**: Identify anomalous articles that are misplaced or not meeting quality standards using a Variational Autoencoder (VAE).

### Dataset Properties
- The extended dataset includes 12701 nodes and 302103 edges, with 2000 new articles (half are anomalous).

### Approach
- **Model**: Variational Autoencoder.
- **Key Features**: 
    - Uses encoder and decoder networks for dimensionality reduction and reconstruction.
    - Implements ELBO (Evidence Lower Bound) for loss calculation, combining reconstruction loss and Kullback-Leibler divergence.

