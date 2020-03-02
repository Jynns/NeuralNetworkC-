#include <iostream>
#include "NeuralNetwork.h"
#include <math.h>

int main() {
	NeuronalesNetz testNet = NeuronalesNetz(2, 1, 1, .3f);
  
	std::vector<std::vector<float>> X{ { 0.1,.1 }, {.2,.2},{ .3,.3 }, {.4,.4},{ .5,.5 }, {.6,.6} };
	std::vector<std::vector<float>> y{ {0.1},{0.2},{0.3},{0.4},{0.5},{0.6} };
  	//just small test if it fits by letting it overfit
	int epochs = 1000;
	for (int i = 0; i < epochs; i++) {
		testNet.train(X, y);
	}
	
	std::vector<float>Test{7, 7};
	testNet.predict(Test);
  	std::cout << "Output " << testNet.getOutputLayer()[0].getValue() << std::endl;
	system("pause");
}
