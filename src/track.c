#include "track.h"
using namespace std;
using namespace cv;

BBOX_tracker::BBOX_tracker(){
	string trackingAlg = "MEDIANFLOW";
	MultiTracker trackers(trackingAlg);
	m_trackers = trackers;
}

void BBOX_tracker::SetObjects(BBOX_list bbox_list){
	m_objects = bbox_list;
}

BBOX_list BBOX_tracker::GetObjects(){
	return m_objects;
}

void BBOX_tracker::CleanObjects(){
	m_objects.clear();
}

void BBOX_tracker::SetROI(Mat roi){
	m_roi = roi;
}

Mat BBOX_tracker::GetROI(){
	return m_roi; 
}

void BBOX_tracker::InitTracker(){
	m_trackers.objects.clear();
	m_trackers.add(m_roi, m_objects);
}

void BBOX_tracker::update(){
	m_trackers.update(m_roi);
	m_objects = m_trackers.objects;
}

