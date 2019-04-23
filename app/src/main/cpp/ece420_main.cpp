//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"
#include <cmath>

#include <unordered_map>

#include <opencv2/ml/ml.hpp>

// JNI Function
extern "C" {
JNIEXPORT int JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_init(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_startAdd(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_doneAdd(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_startAddBackground(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_doneAddBackground(JNIEnv *env, jclass);
}

extern "C" {
JNIEXPORT int JNICALL
Java_com_ece420_lab4_MainActivity_getCurrentSpeaker(JNIEnv *env, jclass);
}

// Student Variables
#define F_S 48000
#define FRAME_SIZE 1024
#define NUM_FILTERS 20

#define BUFFER_SIZE 50

#define VOICED_THRESHOLD 10000000000  // Find your own threshold
#define RECOGNIZED_THRESHOLD 10

#define SAMPLE_UNVOICED (-1)
#define NO_SPEAKERS (-2)
#define UNRECOGNIZED_SPEAKER (-3)
#define BACKGROUND (-4)

int lastFreqDetected = -1;

kiss_fft_cpx in[FRAME_SIZE];
kiss_fft_cpx out[FRAME_SIZE];

int hist_buff[BUFFER_SIZE];

unsigned hist_idx;

cv::Ptr<cv::ml::KNearest> knn;

std::vector<int> labels;
std::vector<float> coeffs;

std::vector<std::vector<double>> fbank;

unsigned nextSpeaker;
bool addingNewSpeaker;
bool recordingBackground;

void ece420ProcessFrame(sample_buf *dataBuf) {

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
        std::vector<float> mfcc = sampleToMFCC(out, fbank, FRAME_SIZE);

        if(recordingBackground){
            coeffs.insert(coeffs.end(), mfcc.begin(), mfcc.end());
            labels.push_back(BACKGROUND);
            lastFreqDetected = BACKGROUND;
            return;
        }

        if(!knn->isTrained()){
            lastFreqDetected = NO_SPEAKERS;
            return;
        }

        cv::Mat_<float> inputFeature(1, VECTOR_DIM, CV_32F);
        cv::Mat_<float> dist(1, 1, CV_32F);
        std::memcpy(inputFeature.data, mfcc.data(), VECTOR_DIM * sizeof(float));
        int classification = (int) knn->findNearest(inputFeature, 1, cv::noArray(), cv::noArray(), dist);
        float classification_dist = dist[0][0];

        if(addingNewSpeaker){
            if(classification == BACKGROUND && classification_dist < RECOGNIZED_THRESHOLD) {
                lastFreqDetected = BACKGROUND;
            } else {
                coeffs.insert(coeffs.end(), mfcc.begin(), mfcc.end());
                labels.push_back(nextSpeaker);
                lastFreqDetected = nextSpeaker;
            }
            return;
        } else {
            if(classification_dist < RECOGNIZED_THRESHOLD)
                hist_buff[hist_idx++] = classification;
            else {
                hist_buff[hist_idx++] = UNRECOGNIZED_SPEAKER;
            }
        }
    } else {
        if(addingNewSpeaker){
            lastFreqDetected = SAMPLE_UNVOICED;
            return;
        }
        else {
            hist_buff[hist_idx++] = SAMPLE_UNVOICED;
        }
    }

    hist_idx %= BUFFER_SIZE;

    std::unordered_map<int, int> dict(BUFFER_SIZE);
    for(unsigned x = 0; x < BUFFER_SIZE; x++){
        dict[hist_buff[x]]++;
    }
    int max_speaker = SAMPLE_UNVOICED;
    int count = dict[SAMPLE_UNVOICED];
    for(auto x = dict.begin(); x != dict.end(); x++){
        if(x->second > count){
            count = x->second;
            max_speaker = x->first;
        }
    }
    lastFreqDetected = max_speaker;
}

JNIEXPORT int JNICALL
Java_com_ece420_lab4_MainActivity_getFreqUpdate(JNIEnv *env, jclass) {
    return lastFreqDetected;
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_init(JNIEnv *env, jclass) {
    LOGD("Received Call to Init");
    knn = cv::ml::KNearest::create();
    knn->setDefaultK(1);
    knn->setIsClassifier(1);

    for(unsigned idx = 0; idx < BUFFER_SIZE; idx++){
        hist_buff[idx] = -1;
    }
    hist_idx = 0;


    nextSpeaker = 0;
    addingNewSpeaker = 0;
    recordingBackground = 0;

    LOGD("Finished training classifier");

    std::vector<double> mel= generateMelPoints(NUM_FILTERS, FRAME_SIZE, F_S);
	std::vector<unsigned> bin = generateBinPoints(mel, FRAME_SIZE, F_S);

	fbank = filter_bank(NUM_FILTERS, FRAME_SIZE, bin);
	LOGD("Finished generating fbank");
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_startAdd(JNIEnv *env, jclass) {
    LOGD("Received call to startAdd");
    addingNewSpeaker = 1;
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_doneAdd(JNIEnv *env, jclass) {
    LOGD("Received call to doneAdd");
    nextSpeaker++;
    addingNewSpeaker = 0;
    for(unsigned idx = 0; idx < BUFFER_SIZE; idx++){
        hist_buff[idx] = BACKGROUND;
    }
    hist_idx = 0;
    updateKNN(labels, coeffs, knn);
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_startAddBackground(JNIEnv *env, jclass) {
    LOGD("Received call to startAddBackground");
    recordingBackground = 1;
}

JNIEXPORT void JNICALL
Java_com_ece420_lab4_MainActivity_doneAddBackground(JNIEnv *env, jclass) {
    LOGD("Received call to doneAddBackground");
    recordingBackground = 0;
    updateKNN(labels, coeffs, knn);
}

JNIEXPORT int JNICALL
Java_com_ece420_lab4_MainActivity_getCurrentSpeaker(JNIEnv *env, jclass) {
    return nextSpeaker;
}