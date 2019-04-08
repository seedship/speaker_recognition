//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include <cmath>
#include "ece420_lib.h"

// https://en.wikipedia.org/wiki/Hann_function
float getHanningCoef(int N, int idx)
{
	return (float) (0.5 * (1.0 - cos(2.0 * M_PI * idx / (N - 1))));
}

int findMaxArrayIdx(float *array, int minIdx, int maxIdx)
{
	int ret_idx = minIdx;

	for (int i = minIdx; i < maxIdx; i++) {
		if (array[i] > array[ret_idx]) {
			ret_idx = i;
		}
	}

	return ret_idx;
}

int findClosestIdxInArray(float *array, float value, int minIdx, int maxIdx)
{
	int retIdx = minIdx;
	float bestResid = abs(array[retIdx] - value);

	for (int i = minIdx; i < maxIdx; i++) {
		if (abs(array[i] - value) < bestResid) {
			bestResid = abs(array[i] - value);
			retIdx = i;
		}
	}

	return retIdx;
}

// TODO: These should really be templatized
int findClosestInVector(std::vector<int> vec, float value, int minIdx, int maxIdx)
{
	int retIdx = minIdx;
	float bestResid = abs(vec[retIdx] - value);

	for (int i = minIdx; i < maxIdx; i++) {
		if (abs(vec[i] - value) < bestResid) {
			bestResid = abs(vec[i] - value);
			retIdx = i;
		}
	}

	return retIdx;
}

float HzToMel(float freq)
{
	return 2595 * log10( (freq / 700.0) + 1);
}

float MelToHz(float mel)
{
	return (pow(10, mel / 2595.0) - 1) * 700;
}

std::vector<std::vector<float>> filter_bank(unsigned num_filters, unsigned nfft, const std::vector<unsigned> &bin_points)
{
	unsigned filter_length = nfft/2 + 1;
	std::vector<std::vector<float>> ans(num_filters);

	for(unsigned j = 0; j < num_filters; j++){
		std::vector<float> subfilter = std::vector<float>(filter_length, 0.0);
		for(auto i = bin_points[j]; i < bin_points[j+1]; i++){
			subfilter[i] = (float)(i-bin_points[j]) / (bin_points[j + 1] - bin_points[j]);
		}
		for(auto i = bin_points[j+1]; i < bin_points[j+2]; i++){
			subfilter[i] = (float)(bin_points[j+2]-i) / (bin_points[j + 2] - bin_points[j+1]);
		}
		ans[j] = subfilter;
	}

	return ans;
}

std::vector<float> generateMelPoints(unsigned num_filters, unsigned nfft, unsigned fs)
{
	float highmel = HzToMel(fs/2.0);

	std::vector<float> ans(num_filters+2);
	ans[0] = 0.0;
	for(unsigned idx = 1; idx < num_filters+2; idx++){
		ans[idx] = ans[idx-1] + (highmel / (num_filters + 1) );
	}
	return ans;
}



std::vector<unsigned> generateBinPoints(const std::vector<float> &mel_points, unsigned nfft, unsigned fs)
{
	std::vector<unsigned> ans(mel_points.size());
	for(unsigned x = 0; x < mel_points.size(); x++){
		ans[x] = (int)((MelToHz(mel_points[x])/fs) * (nfft + 1));
	}
	return ans;
}

bool isVoiced(kiss_fft_cpx *data, unsigned length, unsigned threshold)
{
	float total = 0;
	for(unsigned idx = 0; idx < length; idx++)
	{
		total += data->r * data->r + data->i * data->i;
	}

	return total > threshold
}
