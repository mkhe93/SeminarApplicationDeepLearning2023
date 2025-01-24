## Approach

The CNN model for handwritten digit classification from the lecture was used as the baseline model ([Part1_MNIST.ipynb](https://github.com/aamini/introtodeeplearning/tree/master/lab2)).
Accordingly, the two-dimensional layers were changed to one-dimensional layers.
In the initial trials, this model achieved only a 90% accuracy on the test data (for 5 epochs).

By adding convolutional layers, the accuracy improved to 93%.

A hyperparameter search was conducted using the `keras-tuner` module.
The best model obtained from this search achieved an accuracy of 95% and a loss of 0.2.

## Best Model Architecture

````python
model = keras.Sequential([
    keras.layers.Input(shape=(4200, 6)),

    keras.layers.Conv1D(filters=64, kernel_size=6, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=5),

    keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu', padding='causal'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=3),
    
    keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
````
