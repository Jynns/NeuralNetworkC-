#pragma once
#include <vector>
#include "Neuron.h"

class NeuronalesNetz
{
private:
	struct Net
	{
		std::vector<Neuron> inputLayer;
		std::vector<Neuron> hiddenLayer;
		std::vector<Neuron> outputLayer;
		std::vector<float> weights;
	};
	float lr = 0;
	Net  net;
	float& getWeight(int fromIndex, int toIndex, int indexLowerLayer);
public:
	NeuronalesNetz(int numInpLayer, int numOutLayer, int numHidLayer, float lr);
	~NeuronalesNetz();
	std::vector<Neuron> & getOutputLayer();
	void train(std::vector<std::vector<float>>& X, std::vector<std::vector<float>> &y);
	void predict(std::vector<float>& X);
	void backpropagate(std::vector<float> & y_true);
};
