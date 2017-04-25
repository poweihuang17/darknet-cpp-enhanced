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
	private:
		MultiTracker m_trackers;
		Mat m_roi;
		BBOX_list m_objects;
	public:
		BBOX_tracker();
		void SetObjects(BBOX_list bbox_list);
		BBOX_list GetObjects();
		void CleanObjects();
		void SetROI(Mat roi);
		Mat GetROI();
		void InitTracker();
		void update();
		double get_wall_time(){
			struct timeval time;
			if (gettimeofday(&time,NULL)){
				return 0;
			}
			return (double)time.tv_sec + (double)time.tv_usec * .000001;
		}
}; 

#endif
