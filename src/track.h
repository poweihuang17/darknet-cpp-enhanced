#ifndef TRACK_H
#define TRACK_H

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>
#ifdef __linux__
#include <sys/time.h>
#endif

using namespace std;
using namespace cv;

typedef vector<Rect2d> BBOX_list; 

class BBOX_tracker{
	public:
		MultiTracker m_trackers;
		Mat m_roi;
		BBOX_list objects;
		BBOX_tracker();
		void SetObjects(BBOX_list bbox_list);
		void CleanObjects();
		void SetROI(Mat roi);
		void InitTracker();
		void update();
		double get_wall_time()
		{
			struct timeval time;
			if (gettimeofday(&time,NULL)){
				return 0;
			}
			return (double)time.tv_sec + (double)time.tv_usec * .000001;
		}
}; 

#endif
