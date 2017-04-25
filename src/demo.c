#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "track.h"
#ifdef __linux__
#include <sys/time.h>
#endif

#define FRAMES 3
#define FREQ 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static IplImage *cur_frame;
static Mat cur_frame_mat;
static IplImage *roi_mid_ipl;
static IplImage *roi_right_ipl;
static IplImage *roi_left_ipl;
static Mat roi_mid_mat;
static Mat roi_right_mat;
static Mat roi_left_mat;
static image roi_mid_img;
static image roi_right_img;
static image roi_left_img;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;
static int frame_id;
BBOX_tracker mid_tracker;
BBOX_tracker right_tracker;
BBOX_tracker left_tracker;

static int demo_index = 0;
vector<Rect2d> bbox_mid_draw;
vector<Rect2d> bbox_right_draw;
vector<Rect2d> bbox_left_draw;
typedef struct detect_thread_data{
	image roi_img;
	Mat roi_mat;
	int x_offset;
	int y_offset;
	int frame_offset;
}det_thread_arg;

image ipl_to_image(IplImage* src);
image get_image_from_stream_demo(CvCapture *cap, IplImage **cur_frame, int keyframe);
vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

IplImage *crop_IplImage(IplImage *src, CvRect roi){
	cvSetImageROI(src, roi);
	IplImage *src_crop = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvCopy(src, src_crop, NULL);
	cvResetImageROI(src);
	return src_crop;
}


void *fetch_in_thread(void *ptr)
{
	cur_frame = cvQueryFrame(cap);
	cur_frame_mat = cvarrToMat(cur_frame);
	roi_mid_ipl = crop_IplImage(cur_frame, cvRect(800, 500, 416, 416));
	roi_right_ipl = crop_IplImage(cur_frame, cvRect(1216, 500, 416, 416));
	roi_left_ipl = crop_IplImage(cur_frame, cvRect(384, 500, 416, 416));
	roi_mid_mat = cvarrToMat(roi_mid_ipl);
	roi_right_mat = cvarrToMat(roi_right_ipl);
	roi_left_mat = cvarrToMat(roi_left_ipl);

	//  keyframe
	if(frame_id % FREQ == 0){
		if (!cur_frame){
			roi_mid_img = make_empty_image(0, 0, 0);
			roi_right_img = make_empty_image(0, 0, 0);
			roi_left_img = make_empty_image(0, 0, 0);
		}
		roi_mid_img = ipl_to_image(roi_mid_ipl);
		roi_right_img = ipl_to_image(roi_right_ipl);
		roi_left_img = ipl_to_image(roi_left_ipl);
	}
    return 0;
}

vector<Rect2d> detect_roi(image roi){
	float nms = .4;
	layer l = net.layers[net.n-1];
	float *X = roi.data;
	network_predict(net, X);
	if(l.type == DETECTION){
		get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
	} else if (l.type == REGION){
		get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0, demo_hier_thresh);
	} else {
		error("Last layer must produce detections\n");
	}
	if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	vector<Rect2d> bboxes = process_detections(roi, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	return bboxes;
}

void *detect_mid_roi_in_thread(void *ptr){
	if((frame_id) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		bbox_mid_draw = detect_roi(roi_mid_img);
		free_image(roi_mid_img);
		//  initial tracker
		mid_tracker = BBOX_tracker();
		mid_tracker.SetObjects(bbox_mid_draw);
		mid_tracker.SetROI(roi_mid_mat);
		mid_tracker.InitTracker();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_mid_draw.size(); i++){
			bbox_mid_draw.at(i).x += 800;
			bbox_mid_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_mid_draw[i], Scalar(0, 255, 0), 2, 1);
		}
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		mid_tracker.SetROI(roi_mid_mat);
		mid_tracker.update();
		bbox_mid_draw = mid_tracker.GetObjects();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_mid_draw.size(); i++){
			bbox_mid_draw.at(i).x += 800;
			bbox_mid_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_mid_draw[i], Scalar(0, 0, 255), 2, 1);
		}
	}
    return 0;
}

void *detect_left_roi_in_thread(void *ptr){
	if((frame_id + 2) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		bbox_left_draw = detect_roi(roi_left_img);
		free_image(roi_left_img);
		//  initial tracker
		left_tracker = BBOX_tracker();
		left_tracker.SetObjects(bbox_left_draw);
		left_tracker.SetROI(roi_left_mat);
		left_tracker.InitTracker();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_left_draw.size(); i++){
			bbox_left_draw.at(i).x += 384;
			bbox_left_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_left_draw[i], Scalar(0, 255, 0), 2, 1);
		}
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		left_tracker.SetROI(roi_left_mat);
		left_tracker.update();
		bbox_left_draw = left_tracker.GetObjects();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_left_draw.size(); i++){
			bbox_left_draw.at(i).x += 384;
			bbox_left_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_left_draw[i], Scalar(0, 0, 255), 2, 1);
		}
	}
    return 0;
}

void *detect_right_roi_in_thread(void *ptr){
	if((frame_id + 1) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		bbox_right_draw = detect_roi(roi_right_img);
		free_image(roi_right_img);
		//  initial tracker
		right_tracker = BBOX_tracker();
		right_tracker.SetObjects(bbox_right_draw);
		right_tracker.SetROI(roi_right_mat);
		right_tracker.InitTracker();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_right_draw.size(); i++){
			bbox_right_draw.at(i).x += 1216;
			bbox_right_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_right_draw[i], Scalar(0, 255, 0), 2, 1);
		}
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		right_tracker.SetROI(roi_right_mat);
		right_tracker.update();
		bbox_right_draw = right_tracker.GetObjects();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_right_draw.size(); i++){
			bbox_right_draw.at(i).x += 1216;
			bbox_right_draw.at(i).y += 500;
			rectangle(cur_frame_mat, bbox_right_draw[i], Scalar(0, 0, 255), 2, 1);
		}
	}
    return 0;
}

double total_frame_cost = 0.;
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier_thresh = hier_thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }
    if(!cap) error("Couldn't connect to webcam.\n");
    layer l = net.layers[net.n-1];
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));
	pthread_t detect_mid_thread, detect_right_thread, detect_left_thread;
	namedWindow("demo", WINDOW_NORMAL);
	resizeWindow("demo", 640, 480);
    double before = get_wall_time();
    while(1){
		fetch_in_thread(0);
		//  create mid roi thread
		if(pthread_create(&detect_mid_thread, 0, detect_mid_roi_in_thread, 0)) 
			error("Thread creation failed");
		//  create right roi thread
		if(pthread_create(&detect_right_thread, 0, detect_right_roi_in_thread, 0)) 
			error("Thread creation failed");
		//  create left roi thread
		if(pthread_create(&detect_left_thread, 0, detect_left_roi_in_thread, 0)) 
			error("Thread creation failed");
		pthread_join(detect_mid_thread, 0);
		pthread_join(detect_right_thread, 0);
		pthread_join(detect_left_thread, 0);
		rectangle(cur_frame_mat, Point(384, 500), Point(800, 916), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame_mat, Point(800, 500), Point(1216, 916), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame_mat, Point(1216, 500), Point(1632, 916), Scalar(255, 255, 0), 2, 1);
		imshow("demo", cur_frame_mat);
		waitKey(1);
		cvReleaseImage(&roi_mid_ipl);
		cvReleaseImage(&roi_left_ipl);
		cvReleaseImage(&roi_right_ipl);
		
		frame_id += 1;
		double after = get_wall_time();
		double frame_cost = after - before;
		total_frame_cost += frame_cost;
		fps = (frame_id + 1.) / (total_frame_cost);
		printf("\033[2J");
		printf("\033[1;1H");
		printf("[frame id:%d]\n", frame_id);
		printf("FPS:%.1f\n", fps);
		cout << "this frame cost " << 1000 * frame_cost << " ms" << endl;
		before = after;
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
    int i;
	vector<Rect2d> tmp_bbox_list;
    for(i = 0; i < num; ++i){
        float prob = probs[i][0];
        if(prob > thresh){
            printf("%s: %.0f%%\n", names[0], prob*100);
            box b = boxes[i];
            int left  = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top   = (b.y - b.h / 2.) * im.h;
            int bot   = (b.y + b.h / 2.) * im.h;
			/* left += 800; */
			/* right += 800; */
			/* top += 500; */
			/* bot += 500; */
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
