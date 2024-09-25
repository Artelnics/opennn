<div align="center">
  <img src="http://www.opennn.net/images/opennn_git_logo.svg">
</div>

[![Build Status](https://travis-ci.org/{ORG-or-USERNAME}/{REPO-NAME}.png?branch=master)](https://travis-ci.org/Artelnics/opennn)

OpenNN is a software library written in C++ for advanced analytics. It implements neural networks, the most successful machine learning method. 

The main advantage of OpenNN is its high performance. 

This library outstands in terms of execution speed and memory allocation. It is constantly optimized and parallelized in order to maximize its efficiency.

Some typical applications of OpenNN are business intelligence (customer segmentation, churn prevention...), health care (early diagnosis, microarray analysis,...) and engineering (performance optimization, predictive maitenance...).

## Key Features
OpenNN is a high-performance, open-source library for deep learning in C++. Some of its key features include:

- **Neural Networks**: Supports a wide range of neural network architectures, including multilayer perceptrons, deep networks, and recurrent neural networks.
- **Training Strategies**: Includes advanced training strategies like gradient descent, conjugate gradient, and Levenberg-Marquardt.
- **Data Handling**: Efficient data handling with built-in functions for normalization, scaling, and splitting datasets.
- **Model Selection**: Helps optimize network architectures with built-in methods for model selection and hyperparameter tuning.
- **Testing & Validation**: Offers tools for evaluating the performance of neural networks using a separate testing dataset.
- **Visualization**: Provides tools for visualizing network architecture, training progress, and performance metrics.

## Quick Start
Here’s a minimal example of how to create and train a neural network using OpenNN:

```cpp
#include <opennn/neural_network.h>
#include <opennn/data_set.h>
#include <opennn/training_strategy.h>

using namespace OpenNN;

int main()
{
    // Define the neural network structure
    NeuralNetwork nn({2, 3, 1});

    // Load dataset (replace with the path to your data)
    DataSet data;
    data.load_data("path_to_data.csv");

    // Configure the training strategy
    TrainingStrategy ts(&nn, &data);
    ts.perform_training();

    // Make predictions
    Matrix<double> inputs = {{0.5, 0.7}, {0.2, 0.8}};
    Matrix<double> outputs = nn.predict(inputs);

    return 0;
}


#### 3. **Performance Tips & Best Practices**
This section provides useful insights into improving performance when using OpenNN, which is not directly available on the website.

```markdown
## Performance Tips & Best Practices
To maximize the performance of your neural networks using OpenNN, consider these best practices:

- **Data Preprocessing**: Normalize and scale your data to ensure faster training and better convergence. OpenNN includes built-in functions to help with this.
- **Batch Training**: Use batch training with large datasets to reduce memory usage and improve training times.
- **Regularization**: Apply techniques like L2 regularization or dropout to prevent overfitting, especially when working with complex models.
- **Learning Rate**: Start with a lower learning rate and adjust as needed. A dynamic learning rate can help achieve faster convergence.
- **Parallel Computing**: OpenNN supports parallel execution for faster training on systems with multiple cores.

## Choosing the Right Neural Network Model
Depending on the nature of your project, different neural network architectures and training strategies may be more appropriate:

- **For Classification Tasks** (e.g., image recognition, text classification):
  - Use a **multilayer perceptron** (MLP) with a softmax output layer for multi-class classification tasks.
  
- **For Time Series Forecasting** (e.g., stock price prediction, weather forecasting):
  - Consider using a **recurrent neural network** (RNN) or **long short-term memory** (LSTM) for sequential data and forecasting tasks.

- **For Regression Problems** (e.g., predicting house prices, continuous outputs):
  - Use a fully connected network with linear activation for regression problems, which map inputs to continuous outputs.

- **Model Selection Tip**: Try multiple architectures and compare performance using OpenNN’s built-in model selection tools.

## Troubleshooting
Here are some common issues you may encounter when using OpenNN and how to resolve them:

- **Issue**: Compilation errors due to missing dependencies.
  - **Solution**: Ensure that you have installed all required dependencies listed on the OpenNN website. For example, ensure you have a compatible C++ compiler and the necessary libraries.

- **Issue**: Slow training times with large datasets.
  - **Solution**: Try batch training, or use the parallelization feature to speed up the process by utilizing multiple cores.

- **Issue**: Neural network not converging during training.
  - **Solution**: Check that your data is properly normalized, and consider adjusting the learning rate or using a different training strategy.

- **Issue**: Out-of-memory errors when dealing with large datasets.
  - **Solution**: Use batch processing to reduce memory usage or split your dataset into smaller chunks.

The documentation is composed by tutorials and examples to offer a complete overview about the library. 

### Note:
This README has been enhanced to provide additional, complementary content that helps developers get started more quickly and understand how to maximize performance when using OpenNN. For more detailed installation instructions and tutorials, please refer to the official [OpenNN website](https://opennn.net). The content added here does not duplicate the information already available on the website, but rather supplements it to improve the developer experience.

CMakeLists.txt are build files for CMake, it is also used by the CLion IDE.

The .pro files are project files for the Qt Creator IDE, which can be downloaded from its <a href="http://www.qt.io" target="_blank">site</a>. Note that OpenNN does not make use of the Qt library. 

OpenNN is developed by <a href="http://artelnics.com" target="_blank">Artelnics</a>, a company specialized in artificial intelligence. 
