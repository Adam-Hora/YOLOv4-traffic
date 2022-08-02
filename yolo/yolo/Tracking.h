#ifndef TRACKING
#define TRACKING

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class BoundingBox 
{
	public:

	Rect box; 
	vector<Point> centerPositions; 
	double diagonalLenght; 
	bool matchFound; 
	bool tracked; 
	int trackID;
	int NoMatch; 
	Point predictedNextPosition;

	BoundingBox(Rect _box, int _trackID); 
	void  predictNextPosition(void); 
};

#endif TRACKING