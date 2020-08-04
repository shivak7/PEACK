#pragma once
#include <vector>
#include <cmath>

class PMath
{
public:
	
	static double PI() {return 3.14159265;}

	static float dist(std::vector<float>& x, std::vector<float>& y)
	{
		return sqrt(powf((x[0] - y[0]), 2) + powf((x[1] - y[1]), 2));
	}

	static float dot(std::vector<float>& x, std::vector<float>& y)
	{
		return (x[0] * y[0]) + (x[1] * y[1]);
	}

	static float norm(std::vector<float>& x)
	{
		return sqrt(powf(x[0], 2) + powf(x[1], 2));
	}

	static std::vector<float> vec(std::vector<float>& x, std::vector<float>& y)
	{
		std::vector<float> ret;

		ret.push_back(x[0] - y[0]);
		ret.push_back(x[1] - y[1]);

		return ret;
	}

	static std::pair<float, float> pair_center(std::vector<float>& x, std::vector<float>& y)
	{
		return std::pair<float, float>((x[0] + y[0]) / 2.0, (x[1] + y[1]) / 2.0);
	}

};
