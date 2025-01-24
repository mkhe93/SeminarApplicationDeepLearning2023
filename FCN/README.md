## Approach

In an FCN, the flatten layer of the CNN is replaced by a GlobalAveragePooling layer, allowing the model to be trained on data with varying lengths. 
However, due to `tensorflow`, the data must be padded to the same length within a batch.

Initial attempts yield an accuracy of 97% and a loss of 0.09 on the test data (for 10 epochs). 
Additionally, the training time is shorter.

A hyperparameter search results in an accuracy of 99% and a loss of 0.01 for the best model.

## Best Model Architecture


````python
model = keras.Sequential([
    keras.layers.Input(shape=(None, 6)),
    
    keras.layers.Conv1D(filters=64, kernel_size=6, activation='relu', padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=3),
    
    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=3),
    
    keras.layers.GlobalAveragePooling1D(),
    
    keras.layers.Dense(units=64,activation='sigmoid'),
    keras.layers.Dense(4, activation='softmax')
])
````