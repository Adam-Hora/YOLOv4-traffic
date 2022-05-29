// This code is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

#include <fstream>
#include <sstream>
#include <iostream>
#include<conio.h> 
#include <iomanip>
#include <chrono> 

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace dnn;
using namespace std;
using namespace cuda;

const Scalar SCALAR_GREEN = Scalar(0, 200, 0);
const Scalar SCALAR_RED = Scalar(0, 0, 255);
int carcount;
vector<Point> pointList;
int trajectoryCount;


vector<string> load_classes() 
{
    vector<string> class_list;
    ifstream ifs("obj.names");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}


const vector<Scalar> colors = { 
    Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0),
    Scalar(238, 123, 158), Scalar(65, 82, 186), Scalar(14, 87, 125), Scalar(193, 56, 44),
    Scalar(230, 45, 209), Scalar(148, 148, 87), Scalar(24, 245, 217), Scalar(14, 87, 125),
    Scalar(33, 154, 135), Scalar(206, 209, 108), Scalar(117, 145, 137), Scalar(155, 129, 244),
    Scalar(53, 61, 6), Scalar(145, 75, 152)
};


void drawCarCountOnImage(int& carCount, Mat& frame) 
{
    double dblFontScale = (frame.rows * frame.cols) / 500000.0;
    int intFontThickness = (int)round(dblFontScale * 1.5);

    Size textSize = getTextSize(to_string(carCount), FONT_HERSHEY_DUPLEX, dblFontScale, intFontThickness, 0);

    Point ptTextBottomLeftPosition;

    ptTextBottomLeftPosition.x = frame.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

    putText(frame, to_string(carCount), ptTextBottomLeftPosition, FONT_HERSHEY_DUPLEX, dblFontScale, SCALAR_GREEN, intFontThickness);
}

int main(int argc, char** argv)
{
    int carmax = 0;
    double caravg = 0;
    int carmaxall = 0;
    double caravgall = 0;
    vector<string> class_list = load_classes();

    Mat frame;
    VideoWriter video;
    VideoCapture capture("test2.mp4");
    if (!capture.isOpened()) 
    {
        cerr << "Error opening video file\n";
        return -1;
    }


    Net net;

    net = readNetFromDarknet("yolo-obj.cfg", "yolo-obj_best.weights");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    auto model = DetectionModel(net);
    model.setInputParams(1. / 255, Size(416, 416), Scalar(), true);

    auto start = chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    static const string WindowName = "YOLO";
    namedWindow(WindowName, WINDOW_NORMAL);

    video.open("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)));
    
    while (true)
    {
        carcount = 0;
        capture.read(frame);
        if (frame.empty())
        {
            cout << "Finished reading file\n";
            break;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        model.detect(frame, classIds, confidences, boxes, 0.5, 0.2);
        frame_count++;
        total_frames++;

        int detections = classIds.size();
        if (detections == 0)
        {
            trajectoryCount = 0;
        }
        for (int i = 0; i < detections; ++i) 
        {
            Point center;
            auto box = boxes[i];
            auto classId = classIds[i];
            auto confidence = confidences[i];
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            ostringstream label_ss;
            label_ss << class_list[classId] << ": " << fixed << setprecision(2) << confidence;
            auto label = label_ss.str();

            int baseline;
			Size labelSize = getTextSize(label, FONT_HERSHEY_DUPLEX, 1, 1, &baseline);
			rectangle(frame, Point(box.x, box.y - labelSize.height - baseline - 10), Point(box.x + labelSize.width, box.y), color, FILLED);
			putText(frame, label, Point(box.x, box.y - baseline - 5), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

            if (classId == 11 || classId == 12 || classId == 13 || classId == 15 || classId == 16) 
            {
                center.x = box.x + box.width / 2;
                center.y = box.y + box.height / 2;
                pointList.push_back(center);
                carcount++;
                caravg++;
                caravgall++;
            }
            if (trajectoryCount < 10)
            {
                for (int j = 0; j < pointList.size(); j++)
                {
                    circle(frame, pointList[j], 5, SCALAR_RED, -1);
                    trajectoryCount++;
                }
            }
            else
            {
                auto arrayEnd = remove(begin(pointList), end(pointList), pointList[0]);
                for (int j = 0; j < pointList.size(); j++)
                {
                    circle(frame, pointList[j], 5, SCALAR_RED, -1);
                }
            }

        }
        drawCarCountOnImage(carcount, frame);

        if (frame_count >= 30) {

            auto end = chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = chrono::high_resolution_clock::now();
        }

        if (fps > 0) {

            ostringstream fps_label;
            fps_label << "FPS: " << fixed << setprecision(2) << fps;
            string fps_label_str = fps_label.str();

            putText(frame, fps_label_str.c_str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, SCALAR_RED, 2);

        }

        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        imshow(WindowName, detectedFrame);

        if (waitKey(1) != -1) 
        {
            capture.release();
            cout << "Exiting model\n";
            break;
        }
        if (carmax < carcount)
        {
            carmax = carcount;
            if (carmaxall < carmax)
                carmaxall = carmax;
        }
        if (total_frames % 1000 == 0)
        {
            caravg = caravg / 1000;
            cout << "Maximal number of vehicles in the last 1000 frames : " << carmax << "\n";
            cout << "Average number of vehicles in the last 1000 frames : " << setprecision(2) << caravg << "\n";
            caravg = 0;
            carmax = 0;
        }
    }
    caravgall = caravgall / total_frames;

    cout << "Total frames: " << total_frames << "\n";
    cout << "Maximal number of vehicles at any given time : " << carmax << "\n";
    cout << "Total average number of vehicles  : " << setprecision(2) << caravgall << "\n";

    return 0;
}