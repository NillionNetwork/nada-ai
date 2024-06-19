# Examples

The following are the currently available examples:

- [Linear Regression](./linear_regression): shows how to run a linear regression using Nada AI
- [Neural Net](./neural_net): shows how to build & run a simple Feed-Forward Neural Net (both linear layers & activations) using Nada AI
- [Complex Model](./complex_model): shows how to build more intricate model architectures using Nada AI. Contains convolutions, pooling operations, linear layers and activations
- [Time Series](./time_series): shows how to run a Facebook Prophet time series forecasting model using Nada AI
- [Spam Detection Demo](./spam_detection): shows how to build a privacy-preserving spam detection model using Nada AI. Contains Logistic Regression, and cleartext TF-IDF vectorization.
- [Multi layer Perceptron Demo](./multi_layer_perceptron): shows how to build a privacy-preserving medical image classification model using Nada AI. Features Convolutional Neural Network logic.

In order to run an example, simply:
1. Navigate to the example folder `cd <EXAMPLE_NAME>`
2. Build the program via `nada build`
3. (Optional) Test the program via `nada test`
4. Run the example / demo. This will either be a Python script you can run via `python3 main.py` or a Jupyter notebook where you can just run the cells.

The Nada program source code is stored in `src/<EXAMPLE_NAME>.py`.
