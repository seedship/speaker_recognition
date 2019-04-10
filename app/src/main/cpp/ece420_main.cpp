//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"
#include <cmath>

#include "codebook.h"
#include <opencv2/ml/ml.hpp>
#include "FastDctFft.hpp"

// JNI Function
extern "C" {
JNIEXPORT float JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_init(JNIEnv *env, jclass);
}

// Student Variables
#define F_S 48000
#define FRAME_SIZE 1024
#define NUM_FILTERS 20

#define VOICED_THRESHOLD 5500000000.0  // Find your own threshold
float lastFreqDetected = -1;

kiss_fft_cpx in[FRAME_SIZE];
kiss_fft_cpx out[FRAME_SIZE];

cv::Ptr<cv::ml::KNearest> knn;

std::vector<std::vector<double>> fbank;

void ece420ProcessFrame(sample_buf *dataBuf) {
    // Keep in mind, we only have 20ms to process each buffer!
    LOGD("Received call to process frame");
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    // Data is encoded in signed PCM-16, little-endian, mono
    //float bufferIn[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        int16_t val =
                ((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);
        in[i].r = (float) (val * ( 0.54 - 0.46 * cos((2 * M_PI * i) / (FRAME_SIZE - 1))));
        in[i].i = 0;
    }

    kiss_fft_cfg fft_cfg = kiss_fft_alloc(FRAME_SIZE, 0, NULL, NULL);

    kiss_fft(fft_cfg, in, out);

    free(fft_cfg);
    for(unsigned x = 0; x < FRAME_SIZE; x++){
        out[x].r = out[x].r * out[x].r + out[x].i * out[x].i;
        out[x].i = 0;
    }

    if(isVoiced(out, FRAME_SIZE, VOICED_THRESHOLD)){
        LOGD("Sample is voiced");
//        sampleToMFCC()
//        std::vector<double> mfcc(20);
//
//        for(unsigned x = 0; x < FRAME_SIZE; x++){
//            out[x].r /= FRAME_SIZE;
//        }
//
//        for(unsigned x = 0; x < fbank.size(); x++){
//            float total = 0;
//            for(unsigned y = 0; y < fbank[0].size(); y++){
//                total += out[y].r * fbank[x][y];
//            }
//            mfcc[x] = log(total + 0.001f);
//        }
//
//        FastDctFft::transform(mfcc);
//        for(unsigned x = 0; x < 20; x++){
//            if(!x)
//                mfcc[x] = mfcc[x] * 2 * sqrt(1.0/(4*20));
//            else {
//                mfcc[x] = mfcc[x] * 2 * sqrt(1.0/(2*20));
//            }
//        }
//
//        cv::Mat_<float> inputFeature(1,VECTOR_DIM);
//        for(unsigned x = 0; x < VECTOR_DIM; x++)
//            inputFeature << mfcc[x];
//        lastFreqDetected = knn->findNearest(inputFeature, 1, cv::noArray());
//        LOGD("Speaker: %f", lastFreqDetected);
//        std::vector<float> mfcc = {4.887655258178710938e+01, -3.205749690532684326e-01, 2.612162828445434570e+00, 1.594911456108093262e+00, 4.888237476348876953e+00, -6.285095810890197754e-01, -5.655811429023742676e-01, 3.386659145355224609e+00, 1.824503839015960693e-01, -4.984181523323059082e-01, -1.591834187507629395e+00, 3.316685259342193604e-01};

        std::vector<float> mfcc = sampleToMFCC(out, fbank, FRAME_SIZE);

        cv::Mat_<float> inputFeature(1, VECTOR_DIM);
//        for(unsigned x = 0; x < VECTOR_DIM; x++)
//            inputFeature << (float)mfcc[x];
        std::memcpy(inputFeature.data, mfcc.data(), VECTOR_DIM * sizeof(float));
        lastFreqDetected = knn->findNearest(inputFeature, 1, cv::noArray());
        LOGD("Speaker: %f", lastFreqDetected);
    } else {
        lastFreqDetected = -1;
    }

    gettimeofday(&end, NULL);
    LOGD("Time delay: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}

JNIEXPORT float JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass) {
    return lastFreqDetected;
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_init(JNIEnv *env, jclass) {
    LOGD("Received Call to Init");
    knn = cv::ml::KNearest::create();
    knn->setDefaultK(1);
    knn->setIsClassifier(1);
//    cv::Mat_<float> trainFeatures(NUM_VECTORS, VECTOR_DIM);
//    for(unsigned x = 0; x < NUM_VECTORS; x++){
//        for(unsigned y = 0; y < VECTOR_DIM; y++){
//            trainFeatures << vectors[x][y];
//        }
//    }
    cv::Mat trainFeatures(NUM_VECTORS, VECTOR_DIM, CV_32F);
    std::memcpy(trainFeatures.data, vectors, NUM_VECTORS * VECTOR_DIM * sizeof(float));

    cv::Mat trainlabels(1,NUM_VECTORS, CV_32F);
    std::memcpy(trainlabels.data, tags, NUM_VECTORS * sizeof(float));
    knn->train(trainFeatures, cv::ml::ROW_SAMPLE, trainlabels);

    LOGD("Finished training classifier");

    std::vector<double> mel= generateMelPoints(NUM_FILTERS, FRAME_SIZE, F_S);
	std::vector<unsigned> bin = generateBinPoints(mel, FRAME_SIZE, F_S);

	fbank = filter_bank(NUM_FILTERS, FRAME_SIZE, bin);
	LOGD("Finished generating fbank");
}