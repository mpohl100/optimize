# This repository is an experiment to automate neural network shaping
The two toughest parts about machine learning have always been:
1) data engineering (transforming data in such a way that neural networks can deal with it)
2) network shaping (making the right choice about size, dimensions, activation functions and order)

This repository aims at 2) by trying out letting a genetic algorithm do the network shaping.

## Implementing a machine learning library in Rust
It is not feasably to implement all possible layer types and activations functions as a start.
That is why we start with the following layer types:
1) dense layers
   
And the following activation functions:
1) Sigmoid
2) Tanh
3) ReLU

## The train executable
This repository offers the binary named "train", which fits the input data to a neural network shaped by your configuration.
It takes the following parameters:
- The model directory:
  - The directory to which the trained neural network shall be saved
  - Leave this directory empty if you start from scratch and use the shape file to define the shape of your neural network
  - If this directory contains a neural network as a start the program reads it from disk and continues the training
- The shape file:
  - This parameter is used to configure the shape of the neural network that shall be trained
  - Please leave this parameter empty if you want to continue training an existing neural network
- The input csv file:
  - This file shall contain all the training samples for the training session
  - The number of columns of this file must match the input dimension of the neural network
- The target csv file:
  - This file shall contain the expected outputs of the training samples in the inout csv file. The training sample and its expected output must match by line number.
  - The number of columns in this file must match the output dimension of the neural network.
  - The numbers of lines in the input csv file and the target csv file must match.

If all the parameters are passed correctly and all semantic checks of the neural network shape and its dimensions pass, training will be performed.

## The evaluate executable
Furthermore a binary named "evaluate" is offered which evaluates the inputs and checks how accurate the neural network is trained.
The output of the program is a precentage how many samples were producing the expected target output.
It offers the following parameters:
- The model directory:
  - This is a mandatory parameter
  - The neural network shape must be valid, i. e. all the dimensions of the layers must match
- The input csv file:
  - It contains the data samples that shall be evaluated
  - The number of columns must match the input dimension of the neural newtwork
- The target csv file:
  - It contains the target outputs of the data samples
  - The number of columns must match the output dimension of the neural network
 
## The internal data format
This library defines its own simplified data storage model for neural networks on disk:
The model directory must contain a file named "shape.yaml". It defines the shape of the neural network that the program expects to find on disk.
The following example covers all possible values you can enter:
```
layers:
- layer_type: !Dense
    input_size: 128
    output_size: 128
  activation: Tanh
- layer_type: !Dense
    input_size: 128
    output_size: 64
  activation: ReLU
- layer_type: !Dense
    input_size: 64
    output_size: 10
  activation: Sigmoid
```
Furthermore a subdirectory named "layers" is expected. There files named layer_0 and layer_1, ..., layer_n are expected to be found.
A layer_* file has the following format:
```
2 3
0.00123 -0.156 1.567
1.89 -1.43 0.067
```
The first line contains the number of rows and columns.
In the following lines all the entries of the matrix of the dense layer shall be present.
          

