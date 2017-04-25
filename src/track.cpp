#include "track.h"
using namespace std;
using namespace cv;

BBOX_tracker::BBOX_tracker(){
	string trackingAlg = "MEDIANFLOW";
	MultiTracker trackers(trackingAlg);
	m_trackers = trackers;
}

void BBOX_tracker::SetObjects(BBOX_list bbox_list){
	objects = bbox_list;
}

void BBOX_tracker::CleanObjects(){
	objects.clear();
}

void BBOX_tracker::SetROI(Mat roi){
	m_roi = roi;
}

void BBOX_tracker::InitTracker(){
	m_trackers.objects.clear();
	m_trackers.add(m_roi, objects);
}

void BBOX_tracker::update(){
	double before = get_wall_time();
	m_trackers.update(m_roi);
	double after = get_wall_time();
	cout << "tracking objects cost " << 1000 * (after - before) << "ms" << endl;
}

