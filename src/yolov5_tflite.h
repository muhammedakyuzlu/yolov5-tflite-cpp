#pragma once

#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

struct Prediction
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
};

class YOLOV5
{
public:
    // Take a model path as string
    void loadModel(const  std::string path);
    // Take an image and return a prediction
    void run(cv::Mat image, Prediction &out_pred);

    void getLabelsName(std::string path, std::vector<std::string> &labelNames);

    // thresh hold
    float confThreshold = 0.5;
    float nmsThreshold = 0.5;

    // number of threads
    int nthreads = 4;

private:
    // model's
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    tflite::StderrReporter _error_reporter;

    // parameters of interpreter's input
    int _input;
    int _in_height;
    int _in_width;
    int _in_channels;
    int _in_type;

    // // parameters of interpreter's output
    // int _out;
    // TfLiteIntArray* _out_dims;
    // int  _out_row;
    // int  _out_colum;
    // int  _out_type;                   =

    // parameters of original image
    int _img_height;
    int _img_width;

    // Input of the interpreter
    uint8_t *_input_8;

    // int _delegate_opt;
    // TfLiteDelegate *_delegate;

    template <typename T>
    void fill(T *in, cv::Mat &src);
    void preprocess(cv::Mat &image);
    std::vector<std::vector<float>> tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum);
    void nonMaximumSupprition(
        std::vector<std::vector<float>> &predV,
        const int &row,
        const int &colum,
        std::vector<cv::Rect> &boxes,
        std::vector<float> &confidences,
        std::vector<int> &classIds,
        std::vector<int> &indices);
};