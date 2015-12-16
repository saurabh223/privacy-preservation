//---> Body Detection and Tracking using HOG Classifiers and Blurring ROI <---//
#include <opencv2\core\core.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2\videostab\videostab.hpp>
#include <opencv2\video\tracking.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main (int argc, const char * argv[])
{
	// Declare Frames and Variables
	Mat current_frame, roi, roi_gray, roi_copy, blurred;
	int number_of_people, i;
	int max_count = 15;

	VideoCapture cap(0); // for the web-cam on the laptop
	//VideoCapture cap(1); // for the external camera 
	cap.set(3, 320);
	cap.set(4, 240);

	// De-size the input feed
	Size sz(320, 240); 
	Size kernal_z(15,15);

	//HOG Descriptor
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//Detection And Tracking Loop
	while (1) 
	{
		cap >> current_frame;

		resize(current_frame, current_frame, sz);
		namedWindow("Original Video",1);
		imshow("Original Video", current_frame);

		// Declare Vectors
		vector<Rect> found, found_filtered;
		vector<Point2f> features_next; //set up feature vector
		
		// Detect People in the current_frame
		hog.detectMultiScale(current_frame, found, 0, Size(8,8), Size(32,32), 1.05, 2);

		size_t i, j;

		// This for loop is used to detect only one body in the video feed. Multiple bodies can be detected if it is commented
		for (i=0; i<found.size(); i++)
		{
			Rect r = found[i];
			for (j=0; j<found.size(); j++)
				if (j!=i && (r & found[j])==r)
					break;
			if (j==found.size())
				found_filtered.push_back(r);
		}

		found_filtered = found;
		for (i=0; i<found_filtered.size(); i++)
		{
			if(found_filtered[i].x>0 && found_filtered[i].y>0 && (found_filtered[i].x+found_filtered[i].width<current_frame.cols) && (found_filtered[i].y+found_filtered[i].height)<current_frame.rows)
			{
				Rect region_of_interest = found_filtered[i];

				// set Mat roi as the region of interest
				roi = current_frame(region_of_interest);
				if(!roi.empty())
				{
					//Crop the region of interest from original feed
					namedWindow("ROI",1);
					imshow("ROI", roi);
				}

				// Draws a Rectangle considering the points on the top left and the bottom right of the rectangle
				region_of_interest.x += cvRound(region_of_interest.width*0.1);
				region_of_interest.width = cvRound(region_of_interest.width*0.8);
				region_of_interest.y += cvRound(region_of_interest.height*0.06);
				region_of_interest.height = cvRound(region_of_interest.height*0.9);

				rectangle(current_frame, region_of_interest.tl(), region_of_interest.br(), Scalar(0,255,0), 2);
			}
		}

		// Output the People Found by drawing Rectangles around them
		namedWindow("Found Body",1);
		imshow("Found Body", current_frame);

		// Print How many People Were Found
		number_of_people = found_filtered.size();
		cout <<"Number of People Found: "<<number_of_people<<endl;

		// Find Good Features and Track the Body
		if(number_of_people > 0)
		{
			if(!roi.empty())
			{
				cvtColor( roi, roi_gray, CV_BGR2GRAY );
				
				//Arguments for goodFeaturesToTrack 
				double qualityLevel = 0.01;
				double minDistance = 5;
				int blockSize = 3;
				bool useHarrisDetector = true;
				double k = 0.04;

				//Obtain Initial Set of Features
				goodFeaturesToTrack( roi_gray, features_next, max_count, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );
			}

			for (i=0;i<features_next.size();i++)
			{
				circle( roi, features_next[i], 4, Scalar(0,0,255), -1, 8, 0 );
			}

			if(!current_frame.empty())
			{
				namedWindow("Body Tracked",1);
				imshow("Body Tracked", current_frame);
			}

			//Blur the region of interest in the original feed
			blur(roi,roi,kernal_z);

			//Show the Blurred Body
			if(!current_frame.empty())
			{
				namedWindow("Body Tracked Blurred",1);
				imshow("Body Tracked Blurred", current_frame);
			}
		}
		waitKey(66);
	}
	return 0;
}
