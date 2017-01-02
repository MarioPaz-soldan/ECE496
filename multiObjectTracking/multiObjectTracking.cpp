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

#include <sstream>
#include <iostream>



/////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{

	VideoCapture cap("translate.mp4"); // open the video file for reading

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	//cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

	//double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	//cout << "Frame per seconds : " << fps << endl;

	namedWindow("Object Tracking", CV_WINDOW_AUTOSIZE); //create a window called "Object Tracking"

	// Set up the detector with default parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 200;

	//// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;

	//// Filter by Convexity
	//params.filterByConvexity = true;
	//params.minConvexity = 0.87;

	//// Filter by Inertia
	//params.filterByInertia = true;
	//params.minInertiaRatio = 0.01;

	// Set up detector with params	
	//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Set up the subtractor with default parameters.
	Ptr<BackgroundSubtractor> subtractor = createBackgroundSubtractorMOG2();
	
	// Set up morphlogical elements
	int morph_size_open = 3;
	int morph_size_close = 7;
	Mat element_open = getStructuringElement(MORPH_RECT, Size(2 * morph_size_open + 1, 2 * morph_size_open + 1), Point(morph_size_open, morph_size_open));
	Mat element_close = getStructuringElement(MORPH_RECT, Size(2 * morph_size_close + 1, 2 * morph_size_close + 1), Point(morph_size_close, morph_size_close));

	// Set up the necessary frames for the RGB image, gray image, and masks
	Mat frame_RGB, frame_gray;
	Mat frame_mask, frame_bin, frame_open, frame_close;
	Mat frame_with_keypoints, frame_with_bound;

	while (1)
	{
		// Read a new frame from video
		bool bSuccess = cap.read(frame_RGB); 

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

		//imshow("Object Tracking", frame); //show the frame in "Object Tracking" window

		// Convert image from RGB to grayscale
		cvtColor(frame_RGB, frame_gray, cv::COLOR_BGR2GRAY);

		// Blur the image to remove noise
		GaussianBlur(frame_gray, frame_gray, Size(5,5), 0, 0 );

		// Background Subtration using a Gaussian Mixture model
		subtractor->apply(frame_gray, frame_mask);

		// Threshhold the subtracted image (subtractor output is non binary)
		//imshow("Object Tracking", frame_mask);
		threshold(frame_mask, frame_bin, 175, 255, 0);

		// Apply morphological operations to the subtracted image
		morphologyEx(frame_bin, frame_open, MORPH_OPEN, element_open);
		morphologyEx(frame_open, frame_close, MORPH_CLOSE, element_close);

		// Find bounding rectangle
		Rect boundRect;
		boundRect = boundingRect(frame_close);

		rectangle(frame_RGB, boundRect.tl(), boundRect.br(), Scalar(0,0,255), 2, 8, 0);

		///// Blob Detection
		//std::vector<KeyPoint> keypoints;
		//detector->detect(frame_close, keypoints);

		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		
		//drawKeypoints(frame_close, keypoints, frame_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("Object Tracking", frame_RGB); //show the frame in "object Tracking" window

		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;

}
/*
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
findContours(frame_close, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
vector<Rect> boundRect(contours.size());

vector<vector<Point> > contours_poly(contours.size());
for (size_t i = 0; i < contours.size(); i++)
{
approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
boundRect[i] = boundingRect(Mat(contours_poly[i]));
}

Mat drawing = Mat::zeros(frame_with_bound.size(), CV_8UC3);

for (size_t i = 0; i< contours.size(); i++)
{
Scalar color = Scalar(0,0,255);
drawContours(drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
//This section reads and displays a JPEG file named "Peppers.jpg"

int main(int argc, const char** argv)
{
	Mat img = imread("Peppers.JPG", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'

	if (img.empty()) //check whether the image is loaded or not
	{
		cout << "Error : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}

	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
	imshow("MyWindow", img); //display the image which is stored in the 'img' in the "MyWindow" window

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

	return 0;
}
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////