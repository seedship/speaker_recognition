//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#ifndef ECE420_LIB_H
#define ECE420_LIB_H

#include <math.h>
#include <vector>
#include <kiss_fft.h>

float getHanningCoef(int N, int idx);
int findMaxArrayIdx(float *array, int minIdx, int maxIdx);
int findClosestIdxInArray(float *array, float value, int minIdx, int maxIdx);
int findClosestInVector(std::vector<int> vector, float value, int minIdx, int maxIdx);

float HzToMel(float freq);
float MelToHz(float mel);

bool isVoiced(kiss_fft_cpx *data, unsigned length, unsigned threshold);

std::vector<std::vector<float>> filter_bank(unsigned num_filters, unsigned nfft, const std::vector<unsigned> & bin_points);

std::vector<float> generateMelPoints(unsigned num_filters, unsigned nfft, unsigned fs);

std::vector<unsigned> generateBinPoints(const std::vector<float> &mel_points, unsigned nfft, unsigned fs);

short calculate_MFCC_frame(unsigned fs, kiss_fft_cpx *data, unsigned length, const std::vector<std::vector<float>>& fbank);

#endif //ECE420_LIB_H
