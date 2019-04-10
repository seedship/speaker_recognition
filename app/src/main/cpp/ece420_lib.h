//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#ifndef ECE420_LIB_H
#define ECE420_LIB_H

#include <math.h>
#include <vector>

#include "kiss_fft/kiss_fft.h"

double getHanningCoef(int N, int idx);
int findMaxArrayIdx(double *array, int minIdx, int maxIdx);
int findClosestIdxInArray(double *array, double value, int minIdx, int maxIdx);
int findClosestInVector(std::vector<int> vector, double value, int minIdx, int maxIdx);

double HzToMel(double freq);
double MelToHz(double mel);

double calculateEnergySquared(const kiss_fft_cpx *data_squared, unsigned length);

bool isVoiced(const kiss_fft_cpx *data_squared, unsigned length, double threshold);

std::vector<std::vector<double>> filter_bank(unsigned num_filters, unsigned nfft, const std::vector<unsigned> & bin_points);

std::vector<double> generateMelPoints(unsigned num_filters, unsigned nfft, unsigned fs);

std::vector<unsigned> generateBinPoints(const std::vector<double> &mel_points, unsigned nfft, unsigned fs);

short calculate_MFCC_frame(unsigned fs, const kiss_fft_cpx *data, unsigned length, const std::vector<std::vector<double>>& fbank);

std::vector<double> sampleToMFCC(const std::vector<double> &inputData, const std::vector<std::vector<double> > &fbank);

std::vector<float> sampleToMFCC(kiss_fft_cpx *data, const std::vector<std::vector<double> > &fbank, unsigned nfft);

std::vector<double> naiveDCT(const std::vector<double> &input);

std::vector<float> naiveDCT(const std::vector<float> &input);

double setLastFreqDetected();

#endif //ECE420_LIB_H
