#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "rtree.h"
#include "track.h"
#ifdef __linux__
#include <sys/time.h>
#include <fstream>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>
#include <mutex>
#endif


#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define FREQ 6
//#define FRAME_BY_FRAME
//#define DEBUG

int NET_DIM = 288;
int roi_left_mid_x = 302;
int roi_left_mid_y = 560;
int roi_left_right_x = roi_left_mid_x + NET_DIM;
int roi_left_right_y = roi_left_mid_y;
int roi_right_mid_x = 1144;
int roi_right_mid_y = 680;
int roi_right_right_x = roi_right_mid_x + NET_DIM;
int roi_right_right_y = roi_right_mid_y;
int roi_up_mid_x = 884;
int roi_up_mid_y = 94;
int roi_up_right_x = roi_up_mid_x + NET_DIM;
int roi_up_right_y = roi_up_mid_y;

vector< vector<rtree_box> > cam;
vector< vector<rtree_box> > cam_backup;
int RTREE_BRANCH = 3;
int GRID_SIZE = 3;

FILE *fp;
vector<Rect2d> map_box;

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float **probs;
static box *boxes;
static network net;
static Mat pre_frame;
static Mat cur_frame;
static Mat frames[2500];
static Mat roi_up_mid_mat;
static Mat roi_up_right_mat;
static Mat roi_right_mid_mat;
static Mat roi_right_right_mat;
static Mat roi_left_mid_mat;
static Mat roi_left_right_mat;
static image roi_up_mid_img;
static image roi_up_right_img;
static image roi_right_mid_img;
static image roi_right_right_img;
static image roi_left_mid_img;
static image roi_left_right_img;
static image cur_frame_img;
static image cur_frame_img_s;

const int LEFT_MID = 0;
const int LEFT_RIGHT = 1;
const int UP_MID = 2;
const int UP_RIGHT = 3;
const int RIGHT_MID = 4;
const int RIGHT_RIGHT = 5;

char model_name[30] = "YOLO V2 model";


static VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0.5;
static float demo_thresh_low = 0.5;
static float demo_hier_thresh = 0;
static int frame_id = 0;
BBOX_tracker left_mid_tracker;
BBOX_tracker up_mid_tracker;
BBOX_tracker right_mid_tracker;
BBOX_tracker left_right_tracker;
BBOX_tracker up_right_tracker;
BBOX_tracker right_right_tracker;
mutex mtx;
int box_num = 0;
char *video_file;
char *weight_file;
typedef struct detect_thread_data{
	image roi_img;
	Mat roi_mat;
	BBOX_tracker *tracker;
	int x_offset;
	int y_offset;
	int frame_offset;
}det_thread_arg;

typedef struct yolo_box{
	float x;
	float y;
	float width;
	float height;
	float prob;
}yolo_rect;

vector<Rect2d> process_detections(image im, int num, float demo_thresh, float demo_thresh_low, box *boxes, float **probs, char **names, image **alphabet, int classes, int frame_offset);
vector<Rect2d> map_detections(vector<Rect2d> bbox_draw, int frame_offset);

bool intersect(Rect2d b1, Rect2d b2){

	if(b1.x+b1.width < b2.x || b1.x > b2.x+b2.width || b1.y > b2.y+b2.height || b1.y+b1.height < b2.y){
		return false;
	}
	else{
		int lx, ly, rx, ry;
		lx = max(b1.x, b2.x);
		ly = max(b1.y, b2.y);
		rx = min(b1.x+b1.width, b2.x+b2.width);
		ry = min(b1.y+b1.height, b2.y+b2.height);

		int intersection = (rx-lx+1)*(ry-ly+1);
		int b1_area = b1.width * b1.height;
		int b2_area = b2.width * b2.height;
		float inter_area1 = (float)intersection / (float) (b1_area);
		float inter_area2 = (float)intersection / (float) (b2_area);
		if(inter_area1 > 0.4 || inter_area2 > 0.4) return true;
		else return false;
	}
}

Rect2d merge(Rect2d b1, Rect2d b2){
	Rect2d box;
	box.x = min(b1.x, b2.x);
	box.y = min(b1.y, b2.y);
	double rx, ry;
	rx = max(b1.x+b1.width, b2.x+b2.width);
	ry = max(b1.y+b1.height, b2.y+b2.height);
	box.width = rx - box.x;
	box.height = ry - box.y;
	return box;
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double cosine_similarity(Mat A, Mat B){
	double ab = A.dot(B);
	double aa = A.dot(A);
	double bb = B.dot(B);
	return -ab / sqrt(aa*bb);
}

int mkpath(char* file_path, mode_t mode) {
	assert(file_path && *file_path);
	char* p;
	for(p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/')) {
		*p = '\0';
		if (mkdir(file_path, mode) == -1) {
			if (errno != EEXIST){ 
				*p = '/'; 
				return -1; 
			}
		}
		*p = '/';
	}
return 0;
}

image mat_to_image(Mat src){

    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    image out = make_image(w, h, c);
    if(!out.data){
        printf("@ mat_to_image, out.data is NULL\n");
        exit(-1);
    }
    int count = 0;;
	for(int i = 0; i < h; ++i){
		unsigned char *srcData = src.ptr<unsigned char>(i);
		for(int j = 0; j < w; ++j){
			out.data[count] = srcData[c*j + 2] / 255.;
			out.data[count + h*w] = srcData[c*j + 1] / 255.;
			out.data[count + h*w*2] = srcData[c*j + 0] / 255.;
			count++;
		}
	}
    return out;
}

void *fetch_in_thread(void *ptr){
	cap >> cur_frame;
	
	roi_left_mid_mat = cur_frame(Rect(roi_left_mid_x, roi_left_mid_y, NET_DIM, NET_DIM));
	roi_left_right_mat = cur_frame(Rect(roi_left_right_x, roi_left_right_y, NET_DIM, NET_DIM));
	roi_right_mid_mat = cur_frame(Rect(roi_right_mid_x, roi_right_mid_y, NET_DIM, NET_DIM));
	roi_right_right_mat = cur_frame(Rect(roi_right_right_x, roi_right_right_y, NET_DIM, NET_DIM));
	roi_up_mid_mat = cur_frame(Rect(roi_up_mid_x, roi_up_mid_y, NET_DIM, NET_DIM));
	roi_up_right_mat = cur_frame(Rect(roi_up_right_x, roi_up_right_y, NET_DIM, NET_DIM));
	
	if(frame_id % FREQ == UP_MID)
		roi_up_mid_img = mat_to_image(roi_up_mid_mat);
	if(frame_id % FREQ == UP_RIGHT)
		roi_up_right_img = mat_to_image(roi_up_right_mat);
	if(frame_id % FREQ == RIGHT_MID)
		roi_right_mid_img = mat_to_image(roi_right_mid_mat);
	if(frame_id % FREQ == RIGHT_RIGHT)
		roi_right_right_img = mat_to_image(roi_right_right_mat);
	if(frame_id % FREQ == LEFT_MID)
		roi_left_mid_img = mat_to_image(roi_left_mid_mat);
	if(frame_id % FREQ == LEFT_RIGHT)
		roi_left_right_img = mat_to_image(roi_left_right_mat);
	//rectangle(cur_frame, Point(0, 552), Point(961, 880), Scalar(0, 0, 0), CV_FILLED, 1);
   
	rectangle(cur_frame, Point(roi_right_mid_x, roi_right_mid_y), Point(roi_right_mid_x + NET_DIM, roi_right_mid_y + NET_DIM), Scalar(0, 0, 0), 2, 1);
	rectangle(cur_frame, Point(roi_right_right_x, roi_right_right_y), Point(roi_right_right_x + NET_DIM, roi_right_right_y + NET_DIM), Scalar(0, 0, 0), 2, 1);
	rectangle(cur_frame, Point(roi_left_mid_x, roi_left_mid_y), Point(roi_left_mid_x + NET_DIM, roi_left_mid_y + NET_DIM), Scalar(0, 0, 0), 2, 1);
	rectangle(cur_frame, Point(roi_left_right_x, roi_left_right_y), Point(roi_left_right_x + NET_DIM, roi_left_right_y + NET_DIM), Scalar(0, 0, 0), 2, 1);
	rectangle(cur_frame, Point(roi_up_mid_x, roi_up_mid_y), Point(roi_up_mid_x + NET_DIM, roi_up_mid_y + NET_DIM), Scalar(0, 0, 0), 2, 1);
	rectangle(cur_frame, Point(roi_up_right_x, roi_up_right_y), Point(roi_up_right_x + NET_DIM, roi_up_right_y + NET_DIM), Scalar(0, 0, 0), 2, 1);

	return 0;
}

vector<Rect2d> detect_roi(image roi, int frame_offset){
	float nms = .4;
	layer l = net.layers[net.n-1];
	float *X = roi.data;
	network_predict(net, X);
	if(l.type == DETECTION){
		get_detection_boxes(l, 1, 1, demo_thresh_low, probs, boxes, 0);
	} else if (l.type == REGION){
		get_region_boxes(l, 1, 1, demo_thresh_low, probs, boxes, 0, 0, demo_hier_thresh);
	} else {
		error("Last layer must produce detections\n");
	}
	if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	
	//process_detections(roi, l.w*l.h*l.n, demo_thresh_low, boxes, probs, demo_names, demo_alphabet, demo_classes, frame_offset);
	vector<Rect2d> bboxes = process_detections(roi, l.w*l.h*l.n, demo_thresh, demo_thresh_low, boxes, probs, demo_names, demo_alphabet, demo_classes, frame_offset);
	
	return bboxes;
}

void *detect_roi_in_thread(void *ptr){
	det_thread_arg arg = *(det_thread_arg *)ptr;
	vector<Rect2d> bbox_draw;
	vector<yolo_rect> bbox_ot;
	Scalar color;
	if((frame_id + arg.frame_offset) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		bbox_draw = detect_roi(arg.roi_img, arg.frame_offset);
		free_image(arg.roi_img);
		//  initial tracker
		*arg.tracker = BBOX_tracker();
		arg.tracker->SetObjects(bbox_draw);
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->InitTracker();
		color = Scalar(255, 0, 127);
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->update();
		bbox_draw = arg.tracker->GetObjects();
		color = Scalar(255, 0, 127);
	}


	mtx.lock();
	//  draw origin boxes
	for(unsigned int i = 0; i < bbox_draw.size(); i++){
		bbox_draw[i].x += arg.x_offset;
		bbox_draw[i].y += arg.y_offset;
		if(bbox_draw[i].x < 961 && bbox_draw[i].y > 552) {
			box_num += 1;
		}
		rectangle(cur_frame, bbox_draw[i], Scalar(0,255,0), 4, 1);
	}

	// map boxes
	vector<Rect2d> tmp_map_box = map_detections(bbox_draw, arg.frame_offset);
	for(unsigned int i = 0; i < tmp_map_box.size(); i++){
		map_box.push_back(tmp_map_box[i]);		
	}
	mtx.unlock();

	return 0;
}

double total_frame_cost = 0.;
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh){
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    //demo_thresh = thresh;
    //demo_hier_thresh = hier_thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
	resize_network(&net, NET_DIM, NET_DIM);
    srand(2222222);
    if(filename){
        printf("video file: %s\n", filename);
        cap = VideoCapture(filename);
    }else{
        cap = VideoCapture(cam_index);
    }
    layer l = net.layers[net.n - 1];
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));
	
	pthread_t detect_left_mid_thread, detect_left_right_thread, detect_right_mid_thread, detect_right_right_thread, detect_up_mid_thread, detect_up_right_thread;
	
	/* namedWindow("demo", WINDOW_NORMAL); */
	/* resizeWindow("demo", 640, 480); */

	vector<rtree_box> list;
	cam.push_back(list);  cam.push_back(list);  cam.push_back(list); cam.push_back(list); cam.push_back(list); cam.push_back(list);
	cam_backup.push_back(list);  cam_backup.push_back(list);  cam_backup.push_back(list); cam_backup.push_back(list);  cam_backup.push_back(list);  cam_backup.push_back(list);

	fp = fopen("1", "w+");


	double before, after, frame_cost = 0;
	int k = 0;
    while(frame_id < 2200){
    	before = get_wall_time();
		printf("\033[2J");
		printf("\033[1;1H");
		printf("[frame id:%d]\n", frame_id);
		//printf("FPS:%.1f\n", fps);
		printf("this frame cost: %.2f ms\n", 1000*frame_cost);
		fetch_in_thread(0);
		/*if(frame_id == 690){
			sprintf(model_name, "%s", "Tunnel model");
			char w_f[100] = {0};
			sprintf(w_f, "yolo_weights/tunnel.weights");
			cout << "change weight file: " << w_f << endl;
			load_weights(&net, w_f);
		}
		if(frame_id == 1860){
			sprintf(model_name, "%s", "YOLO v2");
			char w_f[100] = {0};
			sprintf(w_f, "yolo_weights/yolo.weights");
			cout << "change weight file: " << w_f << endl;
			load_weights(&net, w_f);
		}*/
		cam[0].clear();
		cam[1].clear();
		cam[2].clear();
		cam[3].clear();
		cam[4].clear();
		cam[5].clear();
		//  create left mid roi thread
		det_thread_arg left_mid_arg;
		left_mid_arg.roi_img = roi_left_mid_img;
		left_mid_arg.roi_mat = roi_left_mid_mat;
		left_mid_arg.x_offset = roi_left_mid_x;
		left_mid_arg.y_offset = roi_left_mid_y;
		left_mid_arg.tracker = &left_mid_tracker;
		left_mid_arg.frame_offset = LEFT_MID;
		pthread_create(&detect_left_mid_thread, 0, detect_roi_in_thread, (void *)&left_mid_arg);
		//  create left right roi thread
		det_thread_arg left_right_arg;
		left_right_arg.roi_img = roi_left_right_img;
		left_right_arg.roi_mat = roi_left_right_mat;
		left_right_arg.x_offset = roi_left_right_x;
		left_right_arg.y_offset = roi_left_right_y;
		left_right_arg.tracker = &left_right_tracker;
		left_right_arg.frame_offset = LEFT_RIGHT;
		pthread_create(&detect_left_right_thread, 0, detect_roi_in_thread, (void *)&left_right_arg);
	  	//  create up mid roi thread
		det_thread_arg up_mid_arg;
		up_mid_arg.roi_img = roi_up_mid_img;
		up_mid_arg.roi_mat = roi_up_mid_mat;
		up_mid_arg.x_offset = roi_up_mid_x;
		up_mid_arg.y_offset = roi_up_mid_y;
		up_mid_arg.tracker = &up_mid_tracker;
		up_mid_arg.frame_offset = UP_MID;
		pthread_create(&detect_up_mid_thread, 0, detect_roi_in_thread, (void *)&up_mid_arg);
		//  create up right roi thread
		det_thread_arg up_right_arg;
		up_right_arg.roi_img = roi_up_right_img;
		up_right_arg.roi_mat = roi_up_right_mat;
		up_right_arg.x_offset = roi_up_right_x;
		up_right_arg.y_offset = roi_up_right_y;
		up_right_arg.tracker = &up_right_tracker;
		up_right_arg.frame_offset = UP_RIGHT;
		pthread_create(&detect_up_right_thread, 0, detect_roi_in_thread, (void *)&up_right_arg);
		//  create right mid roi thread
		det_thread_arg right_mid_arg;
		right_mid_arg.roi_img = roi_right_mid_img;
		right_mid_arg.roi_mat = roi_right_mid_mat;
		right_mid_arg.x_offset = roi_right_mid_x;
		right_mid_arg.y_offset = roi_right_mid_y;
		right_mid_arg.tracker = &right_mid_tracker;
		right_mid_arg.frame_offset = RIGHT_MID;
		pthread_create(&detect_right_mid_thread, 0, detect_roi_in_thread, (void *)&right_mid_arg);
		//  create right right roi thread
		det_thread_arg right_right_arg;
		right_right_arg.roi_img = roi_right_right_img;
		right_right_arg.roi_mat = roi_right_right_mat;
		right_right_arg.x_offset = roi_right_right_x;
		right_right_arg.y_offset = roi_right_right_y;
		right_right_arg.tracker = &right_right_tracker;
		right_right_arg.frame_offset = RIGHT_RIGHT;
		pthread_create(&detect_right_right_thread, 0, detect_roi_in_thread, (void *)&right_right_arg);

		//// draw black box
		rectangle(cur_frame, Point(0,552), Point(961, 900), Scalar(0,0,0), -1, 1);

		pthread_join(detect_left_mid_thread, 0);
		pthread_join(detect_left_right_thread, 0);
		pthread_join(detect_up_mid_thread, 0);
		pthread_join(detect_up_right_thread, 0);
		pthread_join(detect_right_mid_thread, 0);
		pthread_join(detect_right_right_thread, 0);


		// merge the box
		/*
		for(unsigned int i = 0; i < map_box.size(); i++){
			for(unsigned int j = 0; j < map_box.size(); j++){
				if(intersect(map_box[i], map_box[j]) == true){
					map_box[i] = merge(map_box[i], map_box[j]);			
				}
			}		
		}

		
		for(unsigned int i = 0; i < map_box.size(); i++){
			if(map_box[i].x < 961 && map_box[i].y > 552)
				box_num += 1;
			rectangle(cur_frame, map_box[i], Scalar(255,0,0), 4, 1);
		}
		*/
		map_box.clear();

		// draw the text
		char Text[30], Text2[30];
		sprintf(Text, "Model name: %s", model_name);
		Point TextPos;
		TextPos.x = 50;
		TextPos.y = 610;
		putText(cur_frame , Text , TextPos , CV_FONT_HERSHEY_SIMPLEX, 1 , CV_RGB(255,255,0), 2);
		
		sprintf(Text2, "Detections: %d", box_num);
		Point TextPos2;
		TextPos2.x = 50;
		TextPos2.y = 660;
		putText(cur_frame , Text2 , TextPos2 , CV_FONT_HERSHEY_SIMPLEX, 1 , CV_RGB(255,255,0), 2);


		// Demorgan
		/*
		initTree(GRID_SIZE);
		//Rtree_solution(cur_frame, cam, cam_backup, RTREE_BRANCH, demo_thresh, roi_left_mid_x, roi_left_mid_y);
		//Grid_solution(cur_frame, cam, GRID_SIZE, demo_thresh, roi_left_mid_x, roi_left_mid_y);
		//Origin_solution(cur_frame, cam, demo_thresh, roi_left_mid_x, roi_left_mid_y);
		//printf("cam1 size: %d, cam2 size: %d, cam3 size: %d\n", cam[0].size(), cam[1].size(), cam[2].size());
		*/

		Mat frame_show = cur_frame(Rect(0, 552, 961, 528));//961,528
		cv::namedWindow("demo", CV_WINDOW_NORMAL);
		imshow("demo", frame_show);
		//imshow("demo", cur_frame);

#ifdef FRAME_BY_FRAME
		int k = waitKey(0);
		if(k == 27){
			fclose(fp);
			break;
		}
#else
		int k = waitKey(1);
		if(k == 27){
			fclose(fp);
			break;
		}
#endif

		frame_id += 1;
		after = get_wall_time();
		frame_cost = after - before;
		fps = (frame_id + 1.) / (total_frame_cost);
		total_frame_cost += frame_cost;
    }
	fclose(fp);
	printf("average time cost = %f\n", 1000*total_frame_cost/300);
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh){
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

vector<Rect2d> process_detections(image im, int num, float demo_thresh, float demo_thresh_low, box *boxes, float **probs, char **names, image **alphabet, int classes, int frame_offset)
{
    int i;
	vector<Rect2d> tmp_bbox_list;
    for(i = 0; i < num; ++i){
        float prob = probs[i][2];
        if(prob > demo_thresh_low){
            //printf("%s: %.0f%%\n", names[2], prob*100);
            box b = boxes[i];
            int left  = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top   = (b.y - b.h / 2.) * im.h;
            int bot   = (b.y + b.h / 2.) * im.h;
			Rect bbox;
			bbox.x = left;
			bbox.y = top;
			bbox.width = right - left;
			bbox.height = bot - top;
			tmp_bbox_list.push_back(bbox);
        }
    }
	return tmp_bbox_list;
}

vector<Rect2d> map_detections(vector<Rect2d> bbox_draw, int frame_offset){

	vector<Rect2d> tmp_bbox_list;

	for(unsigned int i = 0; i < bbox_draw.size(); i++){
		int left = bbox_draw[i].x;
		int right = bbox_draw[i].x + bbox_draw[i].width;
		int top = bbox_draw[i].y;
		int bot = bbox_draw[i].y + bbox_draw[i].height;

		if(frame_offset == LEFT_MID){
			//left mid
			//left to left
		}
		else if(frame_offset == LEFT_RIGHT){
			//left right
			//left to left
		}
		else if(frame_offset == UP_MID){
			//up mid
			//up to left
			float a1 = 2.2;
			int b1 = 59;
			left = (left - b1)/a1;
			right = (right - b1)/a1;

			float a2 = 2.2;
			int b2 = -1357;
			top = (top - b2)/a2;
			bot = (bot - b2)/a2;
		}
		else if(frame_offset == UP_RIGHT){
			//up right
			//up to left
			float a1 = 2.2;
			int b1 = 86;
			left = (left - b1)/a1;
			right = (right - b1)/a1;

			float a2 = 2.2;
			int b2 = -1346;
			top = (top - b2)/a2;
			bot = (bot - b2)/a2;
		}
		else if(frame_offset == RIGHT_MID){
			//right mid
			//right to left
			float a1 = 5;
			int b1 = -977;
			left = (left - b1)/a1;
			right = (right - b1)/a1;

			float a2 = 4.5;
			int b2 = -2365;
			top = (top - b2)/a2;
			bot = (bot - b2)/a2;
		}
		else if(frame_offset == RIGHT_RIGHT){
			//right right
			//right to left
			float a1 = 5;
			int b1 = -931;
			left = (left - b1)/a1;
			right = (right - b1)/a1;

			float a2 = 4.5;
			int b2 = -2386;
			top = (top - b2)/a2;
			bot = (bot - b2)/a2;
		}


		//check if the box is in the M
		if (right < roi_left_mid_x || left > roi_left_mid_x + 2 * NET_DIM){}
		else{

			//clip the box
			if (left < roi_left_mid_x) left = roi_left_mid_x;
			if (right > roi_left_mid_x + 2 * NET_DIM) right = roi_left_mid_x + 2 * NET_DIM;
			if (top < roi_left_mid_y) top = roi_left_mid_y;
			if (bot > roi_left_mid_y + 2 * NET_DIM) bot = roi_left_mid_y;
			
			Rect bbox;
			bbox.x = left;
			bbox.y = top;
			bbox.width = right - left;
			bbox.height = bot - top;
			tmp_bbox_list.push_back(bbox);
		}
	}
	return tmp_bbox_list;
}

