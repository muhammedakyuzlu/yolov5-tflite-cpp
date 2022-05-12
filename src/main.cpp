#include "yolov5_tflite.h"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "\nError! Usage: <path to tflite model> <path to classes names> <path to input video> <path to output video>\n"
                  << std::endl;
        return 1;
    }

    Prediction out_pred;
    const std::string model_path = argv[1];
    const std::string names_path = argv[2];
    const std::string video_path = argv[3];
    const std::string save_path = argv[4];

    std::vector<std::string> labelNames;

    YOLOV5 model;

    // conf
    model.confThreshold = 0.30;
    model.nmsThreshold = 0.40;
    model.nthreads = 4;

    // Load the saved_model
    model.loadModel(model_path);

    // Load names
    model.getLabelsName(names_path, labelNames);

    std::cout << "\nLabel Count: " << labelNames.size() << "\n"
              << std::endl;

    cv::VideoCapture capture;
    if (all_of(video_path.begin(), video_path.end(), ::isdigit) == false)
        capture.open(video_path); 
    else
        capture.open(stoi(video_path));

    cv::Mat frame;
    if (!capture.isOpened())
        throw "\nError when reading video steam\n";
    // cv::namedWindow("w", 1);

    // save video config
    bool save = true;
    auto fourcc = capture.get(cv::CAP_PROP_FOURCC);
    int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    cv::VideoWriter video(save_path, fourcc, 30, cv::Size(frame_width, frame_height), true);

    for (;;)
    {
        capture >> frame;
        // frame = cv::imread("bus.jpg");
        if (frame.empty())
            break;
        // start
        auto start = std::chrono::high_resolution_clock::now();
        // Predict on the input image
        model.run(frame, out_pred);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "\nModel run time 'milliseconds': " << duration.count() << "\n"
                  << std::endl;

        // add the bbox to the image and save it
        auto boxes = out_pred.boxes;
        auto scores = out_pred.scores;
        auto labels = out_pred.labels;

        for (int i = 0; i < boxes.size(); i++)
        {
            auto box = boxes[i];
            auto score = scores[i];
            auto label = labels[i];
            cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, labelNames[label], cv::Point(box.x, box.y), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::imshow("output", frame);
        out_pred = {};
        if (save == false)
        {
            cv::resize(frame, frame, cv::Size(frame_width, frame_height), 0, 0, true);
            video.write(frame);
            // cv::imwrite("/home/muhammed/Documents/work/repositories/tflite_c++/test/media/out.jpg",frame);
        }
        cv::waitKey(10);
    }
    capture.release();

    if (save == true)
    {
        video.release();
    }
    cv::destroyAllWindows();
}
