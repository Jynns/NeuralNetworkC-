#include "NeuralNetwork.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>


float & NeuronalesNetz::getWeight(int fromIndex, int toIndex, int indexLowerLayer)
{
	//indexLowerLayer either 0 or 1
	int index = (fromIndex * net.outputLayer.size() + toIndex)*(indexLowerLayer==0) + (net.hiddenLayer.size()*net.inputLayer.size() + net.hiddenLayer.size()*fromIndex+toIndex ) * (indexLowerLayer == 1) ;
	float &weight = net.weights[index];
	return weight;
}

NeuronalesNetz::NeuronalesNetz(int numInpLayer,  int numHidLayer, int numOutLayer, float lr) : lr{lr}
{
	net.inputLayer.reserve(numInpLayer);
	net.hiddenLayer.reserve(numHidLayer);
	net.outputLayer.reserve(numOutLayer);
	net.weights.reserve(numHidLayer * (numOutLayer + numInpLayer));

	net.inputLayer.resize(numInpLayer);
	net.hiddenLayer.resize(numHidLayer);
	net.outputLayer.resize(numOutLayer);
	net.weights.resize(numHidLayer * (numOutLayer + numInpLayer));
	
	//init weights
	srand(time(NULL));


	for (auto weight = 0; weight < net.weights.size(); weight++) {
		net.weights[weight] = (float)(rand() % (RAND_MAX-1) + 1.f)/RAND_MAX;
	}
}

NeuronalesNetz::~NeuronalesNetz()
{
}

std::vector<Neuron> & NeuronalesNetz::getOutputLayer()
{
	return net.outputLayer;
}

void NeuronalesNetz::train(std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& y)
{
	for (int i = 0; i < X.size(); i++) {
		predict(X[i]);
		std::vector<float> y_true{ y[i] };
		backpropagate(y_true);
	}
}

void NeuronalesNetz::predict(std::vector<float>& X)
{
	if (X.size() != net.inputLayer.size()) {
		std::cout << "Data is not aligned \n excpected: " << net.inputLayer.size() << "\n datasize: " << X.size() << std::endl;
	}
	//Input Data
	for (int i = 0; i < X.size(); i++) {
		net.inputLayer[i].setValue(X[i]);
	}
	//From input to hidden

	for (int k = 0; k < net.hiddenLayer.size(); k++) {
		float sumInputs=0;
		for (int i = 0; i < net.inputLayer.size();i++) {
			float weight = getWeight(i, k, 0);
			float test = net.inputLayer[i].getValue() * getWeight(i, k, 0);
			sumInputs += test;
		}
		net.hiddenLayer[k].setValue(sumInputs);
	}
	//From hidden to output
	for (int k = 0; k < net.outputLayer.size(); k++) {
		float sumInputs = .0;
		for (int i = 0; i < net.hiddenLayer.size(); i++) {
			sumInputs += net.hiddenLayer[i].getValue() * getWeight(i, k, 1);
		}
		net.outputLayer[k].setValue(sumInputs);
	}
}

void NeuronalesNetz::backpropagate(std::vector<float>& y_true)
{
	if (y_true.size() != net.outputLayer.size()) { std::cout << "error " << std::endl; return;  }//TODO More output
	//Update weight between hidden output
	for (int i = 0; i < net.outputLayer.size(); i++) {
		for (int c = 0; c < net.hiddenLayer.size(); c++) {
			float outOut = net.outputLayer[i].getValue();
			float outHid = net.hiddenLayer[c].getValue();
			//this value is saved so that it mustnt be recalculated later on
			net.outputLayer[i].setDerivative(-(y_true[i] - outOut)*outOut*(1 - outOut));
			float delta = net.outputLayer[i].getDerivative() *outHid;
			getWeight(c, i, 1) = getWeight(c, i, 1)-lr * delta;
		}
	}
	//Update weight between input hidden
	for (int i = 0; i < net.hiddenLayer.size(); i++) {
		for (int c = 0; c < net.inputLayer.size(); c++) {
			float outHid = net.hiddenLayer[i].getValue();
			float outInp = net.inputLayer[c].getValue();
			float totalError = 0.f;
			//gets too high
			for (int k = 0; k < net.outputLayer.size(); k++) {
				totalError += net.outputLayer[k].getDerivative()*getWeight(i,k,1);
			}
			float delta = outInp *	outHid*(1-outHid)*totalError;
			getWeight(c, i, 0) = getWeight(c, i, 0) - lr * delta;
		}
	}
}
