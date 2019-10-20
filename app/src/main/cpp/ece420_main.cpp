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

#define VOICED_THRESHOLD 10000000000.0  // Find your own threshold
#define RECOGNIZED_THRESHOLD 13

#define SAMPLE_UNVOICED (-1)
#define NO_SPEAKERS (-2)
#define UNRECOGNIZED_SPEAKER (-3)
#define BACKGROUND (-4)


//Variable passed to Java callback
int lastFreqDetected = -1;

//Used to hold FFT
kiss_fft_cpx in[FRAME_SIZE];
kiss_fft_cpx out[FRAME_SIZE];

//Used to average past classifications to prevent noisy readings
int hist_buff[BUFFER_SIZE];
//Circular buffer index
unsigned hist_idx;

//K Nearest neighbor classifier
cv::Ptr<cv::ml::KNearest> knn;

//Running codebook of all the speaker data. Each time a speaker is added, these vectors grow bigger
std::vector<int> labels;
std::vector<float> coeffs;

//Filter bank to perform MFCC computation
std::vector<std::vector<double>> fbank;

//Integer identifier of next speaker
unsigned nextSpeaker;
//Mode flags, for adding new speaker or recording backgrouind
bool addingNewSpeaker;
bool recordingBackground;

void ece420ProcessFrame(sample_buf *dataBuf) {

	// Data is encoded in signed PCM-16, little-endian, mono
	// Unpacking the data and performing Hamming window
	for (int i = 0; i < FRAME_SIZE; i++) {
		int16_t val =
			((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);
		in[i].r = (float) (val * ( 0.54 - 0.46 * cos((2 * M_PI * i) / (FRAME_SIZE - 1))));
		in[i].i = 0;
	}

	//Perform the FFT on the windowed signal
	kiss_fft_cfg fft_cfg = kiss_fft_alloc(FRAME_SIZE, 0, NULL, NULL);
	kiss_fft(fft_cfg, in, out);
	free(fft_cfg);

	//Square the frequency spectrum data, converting to real
	for(unsigned x = 0; x < FRAME_SIZE; x++){
		out[x].r = out[x].r * out[x].r + out[x].i * out[x].i;
	out[x].i = 0;
	}

	//If isVoiced, perform appropriate codebook logging or recognition operation
	if(isVoiced(out, FRAME_SIZE, VOICED_THRESHOLD)){
		//Calculate the MFCC
		std::vector<float> mfcc = sampleToMFCC(out, fbank, FRAME_SIZE);

		if(recordingBackground){
			//Always add to codebook if recording background noise
			coeffs.insert(coeffs.end(), mfcc.begin(), mfcc.end());
			labels.push_back(BACKGROUND);
			lastFreqDetected = BACKGROUND;
			return;
		}

		if(!knn->isTrained()){
			//If no speakers in codebook, return
			lastFreqDetected = NO_SPEAKERS;
			return;
		}

		//Perform classification
		cv::Mat_<float> inputFeature(1, VECTOR_DIM, CV_32F);
		cv::Mat_<float> dist(1, 1, CV_32F);
		std::memcpy(inputFeature.data, mfcc.data(), VECTOR_DIM * sizeof(float));
		int classification = (int) knn->findNearest(inputFeature, 1, cv::noArray(), cv::noArray(), dist);
		float classification_dist = dist[0][0];

		if(addingNewSpeaker){
			//If adding new speaker, but sample matches background, do not add to codebook
			if(classification == BACKGROUND && classification_dist < RECOGNIZED_THRESHOLD) {
				lastFreqDetected = BACKGROUND;
			} else {
				//otherwise, append to codebook
				coeffs.insert(coeffs.end(), mfcc.begin(), mfcc.end());
				labels.push_back(nextSpeaker);
				lastFreqDetected = nextSpeaker;
			}
			//When adding new speaker, either set detected speaker to BACKGROUND or the new speaker
			return;
		} else {
			//Otherwise, perform classification, and add it to the buffer to be averaged.
			if(classification_dist < RECOGNIZED_THRESHOLD)
				hist_buff[hist_idx++] = classification;
			else {
				hist_buff[hist_idx++] = UNRECOGNIZED_SPEAKER;
			}
		}
	} else {
		if(addingNewSpeaker | recordingBackground){
			//If the sample was unvoiced and we are in add new speaker mode, set the tablet display to unvoiced this frame
			lastFreqDetected = SAMPLE_UNVOICED;
			return;
		}
		else {
			//If the sample was not unvoiced and we are not in adding data mode, add UNVOICED to the buffer to be averaged
			hist_buff[hist_idx++] = SAMPLE_UNVOICED;
		}
	}
	//wrap the index
	hist_idx %= BUFFER_SIZE;

	//Compute the count of all the classifications in the buffer
	std::unordered_map<int, int> dict(BUFFER_SIZE);
	for(unsigned x = 0; x < BUFFER_SIZE; x++){
		dict[hist_buff[x]]++;
	}
	int max_speaker = SAMPLE_UNVOICED;
	int count = dict[SAMPLE_UNVOICED];
	//Choose the classification this frame to be the most commpn element in the buffer
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

JNIEXPORT void JNICALL //This function is called once upon program startup
Java_com_ece420_lab4_MainActivity_init(JNIEnv *env, jclass) {
	LOGD("Received Call to Init");
	//Initialize a blank KNN classifier
	knn = cv::ml::KNearest::create();
	knn->setDefaultK(1);
	knn->setIsClassifier(1);

	//Initialize history buffer to UNVOICED
	for(unsigned idx = 0; idx < BUFFER_SIZE; idx++){
		hist_buff[idx] = SAMPLE_UNVOICED;
	}
	hist_idx = 0;

	//Initialize flags to 0.
	nextSpeaker = 0;
	addingNewSpeaker = 0;
	recordingBackground = 0;

	LOGD("Finished training classifier");

	//Generate filter bank
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
