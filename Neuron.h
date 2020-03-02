#pragma once
class Neuron
{
private:
	float value;
	float activate();
	float derivative;
public:
	void addValue(float m);
	void setValue(float m);
	float getValue();
	void setDerivative(float m);
	float getDerivative();

	Neuron();
	~Neuron();
};
