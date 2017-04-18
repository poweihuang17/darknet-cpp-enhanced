#include "track.h"
using namespace std;
using namespace cv;

BBOX_tracker::BBOX_tracker(){
	string trackingAlg = "KCF";
	MultiTracker trackers(trackingAlg);
	m_trackers = trackers;
}

void BBOX_tracker::SetObjects(BBOX_list bbox_list){
	objects = bbox_list;
}

void BBOX_tracker::CleanObjects(){
	objects.clear();
}

void BBOX_tracker::SetFrame(IplImage *current_frame){
	cur_frame_mat = cvarrToMat(current_frame);
}

void BBOX_tracker::InitTracker(){
	m_trackers.objects.clear();
	m_trackers.add(cur_frame_mat, objects);
}

void BBOX_tracker::update(){

	// for(int i = 0; i < m_trackers.objects.size(); i++){
		// cout << "objects" << endl;
		// cout << m_trackers.objects[i].x << endl;
		// cout << m_trackers.objects[i].y << endl;
		// cout << m_trackers.objects[i].width << endl;
		// cout << m_trackers.objects[i].height << endl;
	// }
	double before = get_wall_time();
	m_trackers.update(cur_frame_mat);
	double after = get_wall_time();
	cout << "tracking objects cost " << 1000 * (after - before) << "ms" << endl;
}

void BBOX_tracker::draw_tracking(){
	for(unsigned int i = 0; i < m_trackers.objects.size(); i++)
		rectangle(cur_frame_mat, m_trackers.objects[i], Scalar(255, 0, 0), 2, 1);
	imshow("demo", cur_frame_mat);
	waitKey(1);
}
