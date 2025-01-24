## Approach

Initial attempts and tests were conducted in the notebook `LSTM.ipynb`. The foundation was provided by the tutorial from MIT [introtodeeplearning/lab1/Part2_Music_Generation.ipynb](https://github.com/aamini/introtodeeplearning/blob/master/lab1/Part2_Music_Generation.ipynb), which was also part of the course material. Initial tests showed that the LSTM, with a common sampling length of 4200, required several hours for one epoch. Therefore, using the SciPy package _scipy.interpolate.interp1d_, all time series were tested with different lengths.

Another interesting architecture is provided by the TensorFlow library: the [ConvLSTM1D-Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D) with scientific background from [Shi X. et al. (2015)](https://arxiv.org/abs/1506.04214v1).

The LSTMs were subsequently investigated by adding Dense and/or Convolutional layers with different parameter settings using the Keras Tuner in the notebooks `search_[...].ipynb`. In the same script, the best models were additionally tested on normalized data (which did not lead to any improvement) and compared in `evaluation_lstm.ipynb` using various criteria such as the Matthews correlation coefficient (MCC) ([related work](https://arxiv.org/abs/2008.05756)) and a custom-defined loss function (more information can be found in the `Metrics` folder).

## Best Model Architecture

| Layer (type)   |      Output Shape      |  Param # |
|----------|-------------|------:|
| Conv1D |  (None, 125, 192) | 4800 |
| MaxPooling1D |  (None, 62, 192) | 0 |
| Conv1D |  (None, 39, 80) | 368720 |
| MaxPooling1D |  (None, 62, 192) | 0 |
| LSTM |  (None, 32) | 14464 |
| Dense |  (None, 48) | 1584 |
| Dropout |  (None, 48) | 0 |
| Dense |  (None, 4) | 196 |