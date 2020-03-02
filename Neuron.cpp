#include "Neuron.h"
#include <math.h>
#include <iostream>
#include <limits>

float Neuron::activate()
{
	return 1 / (1 + exp(-value));
}

void Neuron::addValue(float m)
{
	value += m;
	if (isnan(value)) {
		std::cout <<value << std::endl;
	}
}

void Neuron::setValue(float m)
{
	value = m;
	if (isnan(m)) {
		std::cout << m<< std::endl;
	}
	activate();
}

float Neuron::getValue()
{
	return value;
}

void Neuron::setDerivative(float m)
{
	derivative = m;
	if (isnan(derivative)) {
		std::cout << value << std::endl;
	}
}

float Neuron::getDerivative()
{
	return derivative;
}

Neuron::Neuron() :derivative{ 0 }, value{0}
{
}

Neuron::~Neuron()
{
}
