// multiObjectTracking.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"

#include <stdbool.h>
#include <sstream>
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	VideoCapture cap(0); // open the video file for reading
	//VideoCapture cap("translate.mp4"); // open the video file for reading

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	//cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
	// Create the window to view the image
	namedWindow("Object Tracking", CV_WINDOW_AUTOSIZE); //create a window called "Object Tracking"

	//// Set up the detector with default parameters.
	//SimpleBlobDetector::Params params;

	//// Change thresholds
	//params.minThreshold = 10;
	//params.maxThreshold = 200;

	//// Filter by Area.
	//params.filterByArea = true;
	//params.minArea = 200;

	////// Filter by Circularity
	////params.filterByCircularity = true;
	////params.minCircularity = 0.1;

	////// Filter by Convexity
	////params.filterByConvexity = true;
	////params.minConvexity = 0.87;

	////// Filter by Inertia
	////params.filterByInertia = true;
	////params.minInertiaRatio = 0.01;

	//// Set up detector with params	
	////Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Set up the subtractor with default parameters.
	Ptr<BackgroundSubtractor> subtractor = createBackgroundSubtractorMOG2();
	
	// Set up morphlogical elements
	int morph_size_open = 5;
	int morph_size_close = 15;
	Mat element_open = getStructuringElement(MORPH_RECT, Size(2 * morph_size_open + 1, 2 * morph_size_open + 1), Point(morph_size_open, morph_size_open));
	Mat element_close = getStructuringElement(MORPH_RECT, Size(2 * morph_size_close + 1, 2 * morph_size_close + 1), Point(morph_size_close, morph_size_close));

	// Set up the necessary frames for the RGB image, gray image, and masks
	Mat frame_cap, frame_RGB, frame_gray;
	Mat frame_mask, frame_bin, frame_open, frame_close;
	Mat frame_with_keypoints, frame_with_bound;

	int counter = 0;

	while (1)
	{
		// Read a new frame from video
		bool bSuccess = cap.read(frame_cap); 
		if (!bSuccess) //if not success, break loop
		{	cout << "Cannot read the frame from video file" << endl;
			break;
		}

		// Take a smaller region to lessen processing time needed
		//Point p1(0, 0);
		//Point p2(1920, 1080);
		Point p1(0, 300);
		Point p2(1920, 400);
		Rect roi(p1.x, p1.y, p2.x, p2.y);
		Mat mask = Mat::zeros(frame_cap.size(), CV_8UC3);
		rectangle(mask, roi, Scalar(255, 255, 255), CV_FILLED, 8, 0);
		bitwise_and(frame_cap, mask, frame_RGB);
		
		// Convert image from RGB to grayscale
		cvtColor(frame_RGB, frame_gray, COLOR_BGR2GRAY);

		// Blur the image to remove noise
		GaussianBlur(frame_gray, frame_gray, Size(5,5), 0, 0 );

		// Background Subtration using a Gaussian Mixture model
		subtractor->apply(frame_gray, frame_mask);

		// Threshhold the subtracted image (subtractor output is non binary)
		threshold(frame_mask, frame_bin, 175, 255, THRESH_BINARY);

		// Apply morphological operations to the subtracted image
		morphologyEx(frame_bin, frame_open, MORPH_OPEN, element_open);
		morphologyEx(frame_open, frame_close, MORPH_CLOSE, element_close);

		//// Find bounding rectangle
		//Rect boundRect = boundingRect(frame_close);
		//rectangle(frame_RGB, boundRect.tl(), boundRect.br(), Scalar(0,0,255), 2, 8, 0);
		
		// Find bounding rotated rectangle
		std::vector<Point> points;
		//Loop over each pixel and create a point
		for (int x = 0; x < frame_close.cols; x++)
			for (int y = 0; y < frame_close.rows; y++)
				if(frame_close.at<uchar>(y,x) > 0)
					points.push_back(Point(x,y));

		RotatedRect bRect = minAreaRect(Mat(points));
		Point2f vtx[4];
		bRect.points(vtx);
		// Draw the bounding box
		for (int i = 0; i < 4; i++)
			line(frame_RGB, vtx[i], vtx[(i + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);

		///// Blob Detection
		//std::vector<KeyPoint> keypoints;
		//detector->detect(frame_close, keypoints);
		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		//drawKeypoints(frame_close, keypoints, frame_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("Object Tracking", frame_RGB); //show the frame in "object Tracking" window
		counter++;
		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}
