
#ifndef ECE420_LIB_H
#define ECE420_LIB_H

#include <math.h>
#include <vector>
#include <opencv2/ml/ml.hpp>

#include "kiss_fft/kiss_fft.h"

#define VECTOR_DIM 20

/**
 * @brief HzToMel - Converts from Hz domain to Mel domain
 * @param freq - frequency in Hz
 * @return - Mel domain frequency
 */
double HzToMel(double freq);

/**
 * @brief HzToMel - Converts from Mel domain to Hz domain
 * @param freq - frequency in Mel
 * @return - Hz domain frequency
 */
double MelToHz(double mel);

/**
 * @brief calculateEnergySquared - Returns squared sample energy of a given input frequency domain sample
 * @param data_squared - kiss_fft_cpx array of FFT samples, after they have been squared
 * @param length - length of array
 * @return - signal energy squared
 */
double calculateEnergySquared(const kiss_fft_cpx *data_squared, unsigned length);

/**
 * @brief isVoiced - Determines if the given sample frequency spectrum is voiced
 * @param data_squared - kiss_fft_cpx array of FFT samples, after they have been squared
 * @param length - length of array
 * @param threshold - Energy squared threshold to be considered voiced
 * @return - 1 if voiced
 */
bool isVoiced(const kiss_fft_cpx *data_squared, unsigned length, double threshold);

/**
 * @brief filter_bank - Generates a filter bank
 * @param num_filters - Number of filters to use
 * @param nfft - Number of points in sample
 * @param bin_points - Bin points for filter bank to use
 * @return Double vector of floats containing filter bank
 */
std::vector<std::vector<double>> filter_bank(unsigned num_filters, unsigned nfft, const std::vector<unsigned> & bin_points);

/**
 * @brief generateMelPoints - Generates Mel points used to create the bin
 * @param num_filters - Number of filters to use
 * @param nfft - Number of points in sample
 * @param fs - Sampling rate
 * @return Vector of Mel points
 */
std::vector<double> generateMelPoints(unsigned num_filters, unsigned nfft, unsigned fs);

/**
 * @brief generateBinPoints - Generated Bin points used to create filter bank
 * @param mel_points - Mel points to be used to generate bin points
 * @param nfft - Number of points in sample
 * @param fs - Sampling rate
 * @return bin points to generate filter bank with
 */
std::vector<unsigned> generateBinPoints(const std::vector<double> &mel_points, unsigned nfft, unsigned fs);

/**
 * @brief sampleToMFCC - Converts input data to MFCC coefficients
 * @param inputData - Time domain input sample
 * @param fbank - filter bank
 * @return vector of MFCC coefficients
 */
std::vector<double> sampleToMFCC(const std::vector<double> &inputData, const std::vector<std::vector<double> > &fbank);

/**
 * @brief sampleToMFCC - Converts input data to MFCC coefficients
 * @param data - Time domain input sample
 * @param fbank - filter bank
 * @return vector of MFCC coefficients
 */
std::vector<float> sampleToMFCC(kiss_fft_cpx *data, const std::vector<std::vector<double> > &fbank, unsigned nfft);

/**
 * @brief naiveDCT - Performs the DCT using the naive n^2 algorithm.
 * @param input - input signal to transform
 * @return - transformed signal
 */
std::vector<double> naiveDCT(const std::vector<double> &input);

/**
 * @brief naiveDCT - Performs the DCT using the naive n^2 algorithm.
 * @param input - input signal to transform
 * @return - transformed signal
 */
std::vector<float> naiveDCT(const std::vector<float> &input);

/**
 * @brief updateKNN - Updates the KNearest to use new training data
 * @param labels - vector of speaker labels
 * @param vectors - 1D vector of MFCC coefficients, expected to be a multiple of VECTOR_DIM
 * @param knn - cv pointer to a K Nearest Neighbors classifier
 */
void updateKNN(const std::vector<int> &labels, const std::vector<float> &vectors, cv::Ptr<cv::ml::KNearest> & knn);

#endif //ECE420_LIB_H
