## Approach

First, the wavelet-transformed data for a base is created through the notebook `create_wavelets.ipynb`.
To save memory, the data is first interpolated to a length of 500.
For the transformation, 20 scales in the interval [1, 500] were chosen, with the distribution exponentially decaying.
The reason for this is that higher frequencies (smaller scales) occur more frequently in the data than lower frequencies.
The number of scales was also chosen due to memory constraints and computation time.

With the data generated in this way, the models can then be trained.

The best results on the models are obtained with the wavelet base `gaus5` among the tested ones (`gaus5`, `morl`, `mexh`).
We achieve an accuracy of 98% and a loss of 0.027 on the CNN. This represents an improvement for the CNN.
The FCN did not show any improvement with the wavelet transformation.

## Best Model Architecture

````python
model = keras.Sequential([
    keras.layers.Input(shape=(20, 500, 6)),

    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D((1, 2)),

    keras.layers.Conv2D(64, 4, activation='relu'),
    keras.layers.MaxPooling2D((1, 3)),

    keras.layers.Conv2D(72, 5, activation='relu'),
    keras.layers.MaxPooling2D((1, 3)),

    keras.layers.Conv2D(80, 6, activation='relu'),
    keras.layers.MaxPooling2D((1, 3)),

    keras.layers.Flatten(),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.7),
    keras.layers.Dense(4, activation='softmax'),
])
````