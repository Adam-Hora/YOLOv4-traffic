// This code is subject to the license terms in the LICENSE file found at http://opencv.org/license.html

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono> 

#include "Tracking.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace cuda;

const Scalar SCALAR_GREEN = Scalar(0, 200, 0);
const Scalar SCALAR_RED = Scalar(0, 0, 255);

//prcission of box matching lower number is more precise, but will misclassify if boxes belong to eachother at higher speed of vhecicles
float precission;

int trackingCount = 0;
vector<Point> pointList[INT16_MAX];
int classeslist[INT16_MAX];

//Loading class names to string vector
vector<string> load_classes() 
{
    vector<string> class_list;
    ifstream ifs("obj.names");
    if (ifs.is_open())
    {
        string line;
        while (getline(ifs, line))
        {
            class_list.push_back(line);
        }
    }
    else
    {
        throw("error");
    }
    return class_list;
}

//Loading parameters from txt file to string vector
vector<string> load_params()
{
    vector<string> param_list;
    ifstream ifs("config.txt");
    if (ifs.is_open())
    {
        string line;
        while (getline(ifs, line))
        {
            stringstream ss(line);
            string text, param;
            getline(ss, text, ' ');
            getline(ss, param, ' ');

            param_list.push_back(param);
        }
    }
    else
    {
        throw("error");
    }
    return param_list;
}

//Vector of colors for all 18 classes
const vector<Scalar> colors = 
{ 
    Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0),
    Scalar(238, 123, 158), Scalar(65, 82, 186), Scalar(14, 87, 125), Scalar(193, 56, 44),
    Scalar(230, 45, 209), Scalar(148, 148, 87), Scalar(24, 245, 217), Scalar(14, 87, 125),
    Scalar(33, 154, 135), Scalar(206, 209, 108), Scalar(117, 145, 137), Scalar(155, 129, 244),
    Scalar(53, 61, 6), Scalar(145, 75, 152)
};

void MatchBoundingBoxes(vector<BoundingBox>& boundingBoxes, vector<BoundingBox>& currentBoxes);
double distanceBetweenCenters(Point point1, Point point2);
void MatchBoxes(vector<BoundingBox>& boundingBoxes, BoundingBox& currentBoundingBox, int& Index);
void AddNewBox(vector<BoundingBox>& boundingboxes, BoundingBox& currentBox);

//Load bounding box coordinates of vehicles into the BoundingBox class
void LoadBoxes(vector<BoundingBox>& boundingBoxes, vector<Rect>& boxes, int classIDs[INT16_MAX], int detections)
{
    vector<BoundingBox> currentBoxes;
    bool first = false;
    if (boundingBoxes.empty())
    {
        first = true;
    }

    for (int i = 0; i < detections; i++)
    {
        if (classIDs[i] == 11 || classIDs[i] == 12 || classIDs[i] == 13 || classIDs[i] == 15 || classIDs[i] == 16)
        {
            if (first == true)
            {
                trackingCount++;
                BoundingBox newBox(boxes[i], trackingCount);
                boundingBoxes.push_back(newBox);
                pointList[newBox.trackID].push_back(newBox.centerPositions[i]);
            }
            else
            {
                BoundingBox currentBox(boxes[i], 0);
                currentBoxes.push_back(currentBox);
            }
        }
    }
    if (first == false)
    {
        MatchBoundingBoxes(boundingBoxes, currentBoxes);
        currentBoxes.clear();
    }
}

//Compares the distance between bounding boxes from previous frame with the bounding box from current frame and decides if the bounding boxes are the same object or new one
void MatchBoundingBoxes(vector<BoundingBox>& boundingBoxes, vector<BoundingBox>& currentBoxes) 
{
    for (auto& boundingbox : boundingBoxes) 
    {
        boundingbox.matchFound = false;

        boundingbox.predictNextPosition();
    }

    for (auto& currentBox : currentBoxes) 
    {
        int Index = 0;
        double ShortestDistance = 50000;

        for (unsigned int i = 0; i < boundingBoxes.size(); i++) 
        {
            if (boundingBoxes[i].tracked == true) 
            {
                double Distance = distanceBetweenCenters(currentBox.centerPositions.back(), boundingBoxes[i].predictedNextPosition);
                if (Distance < ShortestDistance) 
                {
                    ShortestDistance = Distance;
                    Index = i;
                }
            }
        }

        if ((ShortestDistance >= 0) && (ShortestDistance < currentBox.diagonalLenght * precission)) 
        {
            MatchBoxes(boundingBoxes, currentBox, Index);
        }
        else 
        {
            AddNewBox(boundingBoxes, currentBox);
        }

    }

    for (auto& boundingbox : boundingBoxes) 
    {
        if (boundingbox.matchFound == false) 
        {
            boundingbox.NoMatch++;
            boundingbox.centerPositions.push_back(boundingbox.predictedNextPosition);
            boundingbox.box.x = boundingbox.centerPositions.back().x - (boundingbox.box.width / 2);
            boundingbox.box.y = boundingbox.centerPositions.back().y - (boundingbox.box.height / 2);
        }
        
        if (boundingbox.NoMatch >= 10) 
        {
            boundingbox.tracked = false;
            pointList[boundingbox.trackID].clear();
        }
    }
}

// Calculate distance between bounding boxes from previous frame with the bounding box from current frame
double distanceBetweenCenters(Point point1, Point point2) 
{
    int x = abs(point1.x - point2.x);
    int y = abs(point1.y - point2.y);
    return (sqrt(pow(x, 2) + pow(y, 2)));
}

// Match bounding box from previous frame to the bounding box from current frame
void MatchBoxes(vector<BoundingBox>& BoundingBoxes, BoundingBox& currentBox, int& Index) 
{
    BoundingBoxes[Index].box = currentBox.box;
    BoundingBoxes[Index].centerPositions.push_back(currentBox.centerPositions.back());
    BoundingBoxes[Index].diagonalLenght = currentBox.diagonalLenght;

    pointList[BoundingBoxes[Index].trackID].push_back(currentBox.centerPositions.back());

    BoundingBoxes[Index].matchFound = true;
    BoundingBoxes[Index].tracked = true;
    BoundingBoxes[Index].NoMatch = 0;
}

// Create new bounding box with new tracking Id and memorize the center positions of that bounding box under the new trackingId
void AddNewBox(vector<BoundingBox>& boundingboxes, BoundingBox& currentBox)
{
    trackingCount++;
    BoundingBox newBox(currentBox.box,trackingCount);
    newBox.matchFound = true;
    boundingboxes.push_back(newBox);
    pointList[trackingCount].push_back(newBox.centerPositions.back());
}

//Draw trajectory of vehicle bounding boxes on screen
void DrawTrajectory(Mat& frame)
{
    for (int i = 0; i <= trackingCount; i++)
    {
        while (pointList[i].size() > 15)
        {
            pointList[i].erase(pointList[i].begin());
        }
        if (pointList[i].size() < 5)
        {
            for (int j = 0; j < pointList[i].size(); j++)
            {
                circle(frame, pointList[i][j], 4, SCALAR_RED, -1);
            }
        }
        else
        {
            for (int j = 0; j < pointList[i].size(); j++)
            {
                circle(frame, pointList[i][j], 4, SCALAR_RED, -1);
            }
            polylines(frame, pointList[i], false, SCALAR_RED);
        }
    }
}

//Draws the current number of cars which appeared on the screen to the top right corner
void DrawCarCount(Mat& frame) 
{
    double fontScale = (frame.rows * frame.cols) / 500000.0;
    int fontThickness = round(fontScale * 1.5);

    Size textSize = getTextSize(to_string(trackingCount), FONT_HERSHEY_DUPLEX, fontScale, fontThickness, 0);

    Point bottomLeft;

    bottomLeft.x = frame.cols - 1 - textSize.width * 1.25;
    bottomLeft.y = textSize.height * 1.25;

    putText(frame, to_string(trackingCount), bottomLeft, FONT_HERSHEY_DUPLEX, fontScale, SCALAR_GREEN, fontThickness);
}

int main(int argc, char** argv)
{
    vector<BoundingBox> boundingboxes;
    double caravg = 0;
    double caravgall = 0;
    vector<string> class_list = load_classes();
    vector<string> param_list = load_params();
    VideoCapture capture;
    Mat frame;
    VideoWriter video;
    int choice = 0;
    bool run = true;
    float confidence, nms;

    //Opens stream based on parameters in config file, if parameters are invalid alows selection of new options
    if (param_list[0] == "0")
    {
        capture.open(0);
        if (!capture.isOpened())
        {
            cout << "Error opening stream\n";
        }
        else
        {
            run = false;
        }
    }
    else
    {
        capture.open(param_list[0]);
        if (!capture.isOpened())
        {
            while (run == true)
            {
                cout << "Error opening file\nselect:\n";

                cout << "1 to stream from camera \n2 to play recorded video \n";
                cin >> choice;
                switch (choice)
                {
                case 1:
                {
                    capture.open(0);
                    if (!capture.isOpened())
                    {
                        cout << "Error opening stream\n";
                    }
                    else
                    {
                        run = false;
                    }
                }
                break;
                case 2:
                {
                    string adress;
                    cout << "Enter video adress: ";
                    cin >> adress;
                    capture.open(adress);
                    if (!capture.isOpened())
                    {
                        cout << "Error opening video file\n";
                    }
                    else
                    {
                        run = false;
                    }

                }
                break;
                default:
                {
                    cout << "Invalid option\n";
                }
                }
            }
            
        }
        else
        {
            run = false;
        }
    }
    //Converts string parameters to float
    precission = stof(param_list[1]);
    while ((precission <= 0) || (precission > 3))
    {
        cout << "Precission  is:" << precission << "\n";
        cout << "Precission is outside of recomended parameters (0-3> enter new value\n";
        cin >> precission;
    }


    confidence = stof(param_list[3]);
    while ((confidence <= 0) || (confidence > 1))
    {
        cout << "Confidence  is:" << confidence << "\n";
        cout << "Confidence is outside of required parameters (0-1> enter new value\n";
        cin >> confidence;
    }

    nms = stof(param_list[4]);
    while ((nms <= 0) || (nms > 1))
    {
        cout << "Non-maximal suppresion is:" << nms << "\n";
        cout << "Non-maximal suppresion is outside of required parameters (0-1> enter new value\n";
        cin >> nms;
    }

    Net net;

    net = readNetFromDarknet("yolo-obj.cfg", "yolo-obj_best.weights");

    if (param_list[2] == "1")
    {
        cout << "Using CUDA\n";
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }
    else
    {
        cout << "Using CPU\n";
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    

    auto model = DetectionModel(net);
    model.setInputParams(1. / 255, Size(416, 416), Scalar(), true);

    auto start = chrono::high_resolution_clock::now();
    auto starts = chrono::high_resolution_clock::now();
    int frame_count = 0;
    int frame_counts = 0;
    float fps = -1;
    int total_frames = 0;

    static const string WindowName = "YOLO";
    namedWindow(WindowName, WINDOW_NORMAL);

    video.open("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)));
    

    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            cout << "Finished reading file\n";
            break;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        //Detecting current bounding boxes from pretrained model
        model.detect(frame, classIds, confidences, boxes, confidence, nms);

        // Parameters to count fps
        frame_count++;
        frame_counts++;
        total_frames++;

        int detections = classIds.size();

        //Drawing bounding boxes on screen
        for (int i = 0; i < detections; ++i) 
        {
            Point center;
            Rect box = boxes[i];
            int classId = classIds[i];
            float confidence = confidences[i];
            const Scalar color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            ostringstream label_os;
            label_os << class_list[classId] << ": " << fixed << setprecision(2) << confidence;
            string label = label_os.str();

            int baseline;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 0.5, &baseline);
			rectangle(frame, Point(box.x, box.y - labelSize.height - baseline - 10), Point(box.x + labelSize.width, box.y), color, FILLED);
			putText(frame, label, Point(box.x, box.y - baseline - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

            if (classId == 11 || classId == 12 || classId == 13 || classId == 15 || classId == 16) 
            {
                caravg++;
                caravgall++;
            }
            classeslist[i] = classId;
        }

        //Calling functions to draw trajectory and car count
        LoadBoxes(boundingboxes, boxes, classeslist, detections);

        DrawTrajectory(frame);

        DrawCarCount(frame);

        if (frame_count >= 30) 
        {
            auto end = chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = chrono::high_resolution_clock::now();
        }

        if (fps > 0) 
        {

            ostringstream fps_label;
            fps_label << "FPS: " << fixed << setprecision(2) << fps;
            string fps_label_str = fps_label.str();

            putText(frame, fps_label_str.c_str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, SCALAR_RED, 2);

        }

        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        imshow(WindowName, detectedFrame);

        auto ends = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(ends - starts).count() >= 10)
        {
            caravg = caravg / frame_counts;
            frame_counts = 0;
            starts = chrono::high_resolution_clock::now();
            cout << "Average number of vehicles in the last 10 seconds : " << setprecision(2) << caravg << "\n\n";
            caravg = 0;
        }
        if (waitKey(1) != -1)
        {
            capture.release();
            cout << "Exiting model\n";
            break;
        }
    }
    caravgall = caravgall / total_frames;

    cout << "Total frames: " << total_frames << "\n";
    cout << "Total average number of vehicles  : " << setprecision(2) << caravgall << "\n";
    cout << "Total number of vehicles : " << trackingCount << "\n";

    return 0;
}