import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
import pandas as pd
# Load CIFAR-10 dataset
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data'], batch['labels']

def load_cifar10(data_dir):
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch_data, batch_labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(batch_data)
        train_labels += batch_labels

    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return train_data, np.array(train_labels), test_data, np.array(test_labels)

# Preprocess CIFAR-10 data
def preprocess_data(train_data, test_data):
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)
    test_data = scaler.transform(test_data.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)
    return train_data, test_data

# Load CIFAR-10 data
data_dir = 'cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)

# Preprocess data
train_data, test_data = preprocess_data(train_data, test_data)

# Flatten the data for KNN
train_data_flat = train_data.reshape(train_data.shape[0], -1)
test_data_flat = test_data.reshape(test_data.shape[0], -1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data_flat, train_labels, test_size=0.2, random_state=42)

n_neighbors_values = [3, 5, 7, 9]
weights_values = ['uniform', 'distance']

best_accuracy = 0
best_knn_model = None

for n_neighbors in n_neighbors_values:
    for weights in weights_values:
        # Initialize KNN classifier with different n_neighbors and weights
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        # Train the KNN classifier
        knn_classifier.fit(X_train, y_train)

        # Evaluate accuracy on the validation set
        accuracy = metrics.accuracy_score(y_val, knn_classifier.predict(X_val))
        print(f'Validation Accuracy with {n_neighbors} neighbors and weights={weights}: {accuracy:.4f}')

        # Update best model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_knn_model = knn_classifier

# Use the best model for predictions
test_predictions = best_knn_model.predict(test_data_flat)

# Evaluate accuracy on the validation set
accuracy = metrics.accuracy_score(y_val, best_knn_model.predict(X_val))
print(f'Validation Accuracy with Best Model: {accuracy:.4f}')


# Make predictions on the test set
test_predictions = knn_classifier.predict(test_data_flat)

# Create a submission file
submission_data = {'key': np.arange(0, len(test_predictions)), '0': test_predictions}

# Save the submission file as CSV
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('knn_submission.csv', index=False)
