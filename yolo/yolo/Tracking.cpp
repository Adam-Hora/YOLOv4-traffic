#include "Tracking.h"

using namespace cv;
using namespace dnn;
using namespace std;


BoundingBox::BoundingBox(Rect _box, int _trackID) 
{
	box.x = _box.x;
	box.y = _box.y;
	box.width = _box.width;
	box.height = _box.height;

	Point currentCenter;
	currentCenter.x = box.x + box.width / 2;
	currentCenter.y = box.y + box.height / 2;

	centerPositions.push_back(currentCenter);

	trackID = _trackID;

	diagonalLenght = sqrt(pow(box.width, 2) + pow(box.height, 2));
	tracked = true;
	matchFound = true;
	NoMatch = 0;
}

//Predicting next position using the last five
void BoundingBox::predictNextPosition(void) 
{
	int NumberOfPositions = centerPositions.size();
	
	if (NumberOfPositions == 1) 
	{

		predictedNextPosition.x = centerPositions.back().x;
		predictedNextPosition.y = centerPositions.back().y;

	}
	else if (NumberOfPositions == 2) 
	{

		int deltaX = centerPositions[1].x - centerPositions[0].x;
		int deltaY = centerPositions[1].y - centerPositions[0].y;

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (NumberOfPositions == 3) 
	{

		int sumOfXChanges = ((centerPositions[2].x - centerPositions[1].x) * 2) + ((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = round(sumOfXChanges / 3);

		int sumOfYChanges = ((centerPositions[2].y - centerPositions[1].y) * 2) + ((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = round(sumOfYChanges / 3);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (NumberOfPositions == 4) 
	{

		int sumOfXChanges = ((centerPositions[3].x - centerPositions[2].x) * 3) + ((centerPositions[3].x - centerPositions[1].x) * 2) +
			((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = round(sumOfXChanges / 6);

		int sumOfYChanges = ((centerPositions[3].y - centerPositions[2].y) * 3) + ((centerPositions[3].y - centerPositions[1].y) * 2) +
			((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = round(sumOfYChanges / 6);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (NumberOfPositions >= 5) 
	{

		int sumOfXChanges = ((centerPositions[NumberOfPositions - 1].x - centerPositions[NumberOfPositions - 2].x) * 4) +
			((centerPositions[NumberOfPositions - 2].x - centerPositions[NumberOfPositions - 3].x) * 3) +
			((centerPositions[NumberOfPositions - 3].x - centerPositions[NumberOfPositions - 4].x) * 2) +
			((centerPositions[NumberOfPositions - 4].x - centerPositions[NumberOfPositions - 5].x) * 1);

		int deltaX = round(sumOfXChanges / 10);

		int sumOfYChanges = ((centerPositions[NumberOfPositions - 1].y - centerPositions[NumberOfPositions - 2].y) * 4) +
			((centerPositions[NumberOfPositions - 2].y - centerPositions[NumberOfPositions - 3].y) * 3) +
			((centerPositions[NumberOfPositions - 3].y - centerPositions[NumberOfPositions - 4].y) * 2) +
			((centerPositions[NumberOfPositions - 4].y - centerPositions[NumberOfPositions - 5].y) * 1);

		int deltaY = round(sumOfYChanges / 10);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else
	{
		throw("Error");
	}
}