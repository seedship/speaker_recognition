
#include <cmath>
#include "ece420_lib.h"
#include <opencv2/ml/ml.hpp>

double HzToMel(double freq)
{
	return 2595 * log10( (freq / 700.0) + 1);
}

double MelToHz(double mel)
{
	return (pow(10, mel / 2595.0) - 1) * 700;
}

std::vector<std::vector<double>> filter_bank(unsigned num_filters, unsigned nfft, const std::vector<unsigned> &bin_points)
{
	unsigned filter_length = nfft/2 + 1;
	std::vector<std::vector<double>> ans(num_filters);

	for(unsigned j = 0; j < num_filters; j++){
		std::vector<double> subfilter = std::vector<double>(filter_length, 0.0);
		for(auto i = bin_points[j]; i < bin_points[j+1]; i++){
			subfilter[i] = (double)(i-bin_points[j]) / (bin_points[j + 1] - bin_points[j]);
		}
		for(auto i = bin_points[j+1]; i < bin_points[j+2]; i++){
			subfilter[i] = (double)(bin_points[j+2]-i) / (bin_points[j + 2] - bin_points[j+1]);
		}
		ans[j] = subfilter;
	}

	return ans;
}

std::vector<double> generateMelPoints(unsigned num_filters, unsigned nfft, unsigned fs)
{
	double highmel = HzToMel(fs/2.0);

	std::vector<double> ans(num_filters+2);
	ans[0] = 0.0;
	for(unsigned idx = 1; idx < num_filters+2; idx++){
		ans[idx] = ans[idx-1] + (highmel / (num_filters + 1) );
	}
	return ans;
}

std::vector<unsigned> generateBinPoints(const std::vector<double> &mel_points, unsigned nfft, unsigned fs)
{
	std::vector<unsigned> ans(mel_points.size());
	for(unsigned x = 0; x < mel_points.size(); x++){
		ans[x] = (int)((MelToHz(mel_points[x])/fs) * (nfft + 1));
	}
	return ans;
}

double calculateEnergySquared(const kiss_fft_cpx *data_squared, unsigned length)
{
	double total = 0;
	for(unsigned idx = 0; idx < length; idx++)
	{
		total += data_squared->r;
	}
	return total;
}

bool isVoiced(const kiss_fft_cpx *data_squared, unsigned length, double threshold)
{
	double total = calculateEnergySquared(data_squared, length);
	return total > threshold;
}

double setLastFreqDetected(){
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn->setDefaultK(1);
	knn->setIsClassifier(1);
	cv::Mat_<double> trainFeatures(2, 1);
	trainFeatures << 1, 2;

	cv::Mat_<int> trainlabels(1,2);
	trainlabels << 1, 2;
	knn->train(trainFeatures, cv::ml::ROW_SAMPLE, trainlabels);

	cv::Mat_<double> testFeature(1,1);
	testFeature << 1.6;
	return knn->findNearest(testFeature, 1, cv::noArray());
}

std::vector<double> sampleToMFCC(const std::vector<double> &inputData, const std::vector<std::vector<double>> &fbank)
{
	unsigned framesize = inputData.size();
	kiss_fft_cpx data[framesize];
	for(unsigned x = 0; x < framesize; x++){
		data[x].r = inputData[x] * (0.54 - 0.46 * cos((2 * M_PI * x) / (framesize - 1)));
		data[x].i = 0;
	}

	kiss_fft_cfg fft_cfg = kiss_fft_alloc(framesize, 0, NULL, NULL);
	kiss_fft(fft_cfg, data, data);
	free(fft_cfg);

	for(unsigned x = 0; x < framesize; x++){
		data[x].r = data[x].r;
		data[x].i = data[x].i;
	}

	for(unsigned x = 0; x < framesize; x++){
		data[x].r = data[x].r * data[x].r + data[x].i * data[x].i;
		data[x].i = 0;
	}

	for(unsigned x = 0; x < framesize; x++){
		data[x].r /= framesize;
	}

	std::vector<double> mfcc(20);
	for(unsigned x = 0; x < fbank.size(); x++){
		double total = 0;
		for(unsigned y = 0; y < fbank[0].size(); y++){
			total += data[y].r * fbank[x][y];
		}
		mfcc[x] = log(total + 0.001f);
	}

	mfcc = naiveDCT(mfcc);
	for(unsigned x = 0; x < 20; x++){
		if(!x)
			mfcc[x] = mfcc[x] * sqrt(1.0/(4*20));
		else {
			mfcc[x] = mfcc[x] * sqrt(1.0/(2*20));
		}
	}

	return mfcc;
}

std::vector<float> sampleToMFCC(kiss_fft_cpx *data, const std::vector<std::vector<double> > &fbank, unsigned nfft)
{

    for(unsigned x = 0; x < nfft; x++){
        data[x].r /= nfft;
    }

    std::vector<float> mfcc(20);
    for(unsigned x = 0; x < fbank.size(); x++){
        float total = 0;
        for(unsigned y = 0; y < fbank[0].size(); y++){
            total += data[y].r * fbank[x][y];
        }
        mfcc[x] = log(total + 0.001f);
    }

    mfcc = naiveDCT(mfcc);
    for(unsigned x = 0; x < 20; x++){
        if(!x)
            mfcc[x] = mfcc[x] * sqrt(1.0f/(4*20));
        else {
            mfcc[x] = mfcc[x] * sqrt(1.0f/(2*20));
        }
    }

    return mfcc;
}

std::vector<double> naiveDCT(const std::vector<double> &input)
{
	unsigned N = input.size();
	std::vector<double> ans(N);
	for(unsigned k = 0; k < N; k++){
		double sum = 0;
		for(unsigned n = 0; n < N; n++){
			sum += 2* input[n] * cos(M_PI * k * (2*n+1)/(2*N));
		}
		ans[k] = sum;
	}
	return ans;
}

std::vector<float> naiveDCT(const std::vector<float> &input)
{
    unsigned N = input.size();
    std::vector<float> ans(N);
    for(unsigned k = 0; k < N; k++){
        double sum = 0;
        for(unsigned n = 0; n < N; n++){
            sum += 2* input[n] * cos(M_PI * k * double(2*n+1)/(2*N));
        }
        ans[k] = sum;
    }
    return ans;
}

void updateKNN(const std::vector<int> & labels, const std::vector<float> & vectors, cv::Ptr<cv::ml::KNearest> &knn)
{
	unsigned num_vectors = (unsigned)labels.size();
	cv::Mat trainFeatures(num_vectors, VECTOR_DIM, CV_32F);
	std::memcpy(trainFeatures.data, vectors.data(), num_vectors * VECTOR_DIM * sizeof(float));

	cv::Mat trainlabels(1,num_vectors, CV_32S);
	std::memcpy(trainlabels.data, labels.data(), num_vectors * sizeof(int));

	knn->train(trainFeatures, cv::ml::ROW_SAMPLE, trainlabels);
}
