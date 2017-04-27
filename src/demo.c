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

#define FREQ 30

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float **probs;
static box *boxes;
static network net;
static Mat cur_frame;
static Mat roi_mid_mat;
static Mat roi_right_mat;
static Mat roi_left_mat;
static image roi_mid_img;
static image roi_right_img;
static image roi_left_img;
static VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;
static int frame_id;
BBOX_tracker mid_tracker;
BBOX_tracker right_tracker;
BBOX_tracker left_tracker;
typedef struct detect_thread_data{
	image roi_img;
	Mat roi_mat;
	BBOX_tracker *tracker;
	int x_offset;
	int y_offset;
	int frame_offset;
}det_thread_arg;

image ipl_to_image(IplImage* src);
vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
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

IplImage *crop_IplImage(IplImage *src, CvRect roi){
	cvSetImageROI(src, roi);
	IplImage *src_crop = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvCopy(src, src_crop, NULL);
	cvResetImageROI(src);
	return src_crop;
}

void *fetch_in_thread(void *ptr){

	double before = get_wall_time();
	cap >> cur_frame;
	double after = get_wall_time();
	cout << "cv query frame cost " << 1000 * (after - before) << " ms" << endl;
	before = after;
	roi_mid_mat = cur_frame(Rect(800, 500, 416, 416));
	roi_right_mat = cur_frame(Rect(1216, 500, 416, 416));
	roi_left_mat = cur_frame(Rect(384, 500, 416, 416));
	after = get_wall_time();
	cout << "crop opencv frame cost " << 1000 * (after - before) << " ms" << endl;
	before = after;
	//  keyframe
	if((frame_id) % FREQ == 0){
		if (!cur_frame.data){
			roi_mid_img = make_empty_image(0, 0, 0);
		}
		roi_mid_img = mat_to_image(roi_mid_mat);
	}
	if((frame_id + 1) % FREQ == 0){
		if (!cur_frame.data){
			roi_right_img = make_empty_image(0, 0, 0);
		}
		roi_right_img = mat_to_image(roi_right_mat);
	}
	if((frame_id + 2) % FREQ == 0){
		if (!cur_frame.data){
			roi_left_img = make_empty_image(0, 0, 0);
		}
		roi_left_img = mat_to_image(roi_left_mat);
	}
	after = get_wall_time();
	cout << "change opencv mat to image cost " << 1000 * (after - before) << " ms" << endl;
    return 0;
}

vector<Rect2d> detect_roi(image roi){
	float nms = .4;
	layer l = net.layers[net.n-1];
	float *X = roi.data;
	double before = get_wall_time();
	network_predict(net, X);
	double after = get_wall_time();
	cout << "prediction cost " << 1000 * (after - before) << " ms" << endl;
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

void *detect_roi_in_thread(void *ptr){
	det_thread_arg arg = *(det_thread_arg *)ptr;
	if((frame_id + arg.frame_offset) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		vector<Rect2d> bbox_draw = detect_roi(arg.roi_img);
		free_image(arg.roi_img);
		//  initial tracker
		*arg.tracker = BBOX_tracker();
		arg.tracker->SetObjects(bbox_draw);
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->InitTracker();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_draw.size(); i++){
			bbox_draw.at(i).x += arg.x_offset;
			bbox_draw.at(i).y += arg.y_offset;
			rectangle(cur_frame, bbox_draw[i], Scalar(0, 255, 0), 2, 1);
		}
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->update();
		vector<Rect2d> bbox_draw = arg.tracker->GetObjects();
		//  draw boxes
		for(unsigned int i = 0; i < bbox_draw.size(); i++){
			bbox_draw.at(i).x += arg.x_offset;
			bbox_draw.at(i).y += arg.y_offset;
			rectangle(cur_frame, bbox_draw[i], Scalar(0, 0, 255), 2, 1);
		}
	}
    return 0;
}

double total_frame_cost = 0.;
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh){
    //skip = frame_skip;
    image **alphabet = load_alphabet();
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
        cap = VideoCapture(filename);
    }else{
        cap = VideoCapture(cam_index);
    }
    layer l = net.layers[net.n - 1];
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));
	pthread_t detect_mid_thread, detect_right_thread, detect_left_thread;
	namedWindow("demo", WINDOW_NORMAL);
	resizeWindow("demo", 640, 480);
    double before = get_wall_time();
    while(1){
		printf("\033[2J");
		printf("\033[1;1H");
		printf("[frame id:%d]\n", frame_id);
		printf("FPS:%.1f\n", fps);
		fetch_in_thread(0);
		if(1){
			//  create mid roi thread
			det_thread_arg mid_arg;
			mid_arg.roi_img = roi_mid_img;
			mid_arg.roi_mat = roi_mid_mat;
			mid_arg.tracker = &mid_tracker;
			mid_arg.x_offset = 800;
			mid_arg.y_offset = 500;
			mid_arg.frame_offset = 0;
			pthread_create(&detect_mid_thread, 0, detect_roi_in_thread, (void *)&mid_arg);
			//  create right roi thread
			det_thread_arg right_arg;
			right_arg.roi_img = roi_right_img;
			right_arg.roi_mat = roi_right_mat;
			right_arg.tracker = &right_tracker;
			right_arg.x_offset = 1216;
			right_arg.y_offset = 500;
			right_arg.frame_offset = 1;
			pthread_create(&detect_right_thread, 0, detect_roi_in_thread, (void *)&right_arg);
			//  create left roi thread
			det_thread_arg left_arg;
			left_arg.roi_img = roi_left_img;
			left_arg.roi_mat = roi_left_mat;
			left_arg.tracker = &left_tracker;
			left_arg.x_offset = 1632;
			left_arg.y_offset = 500;
			left_arg.frame_offset = 2;
			pthread_create(&detect_left_thread, 0, detect_roi_in_thread, (void *)&left_arg);
			pthread_join(detect_mid_thread, 0);
			pthread_join(detect_right_thread, 0);
			pthread_join(detect_left_thread, 0);
		}else{
			detect_roi_in_thread(0);
			detect_roi_in_thread(0);
			detect_roi_in_thread(0);
		}
		rectangle(cur_frame, Point(384, 500), Point(800, 916), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame, Point(800, 500), Point(1216, 916), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame, Point(1216, 500), Point(1632, 916), Scalar(255, 255, 0), 2, 1);
		/* imshow("demo", cur_frame); */
		/* waitKey(1); */
		frame_id += 1;
		double after = get_wall_time();
		double frame_cost = after - before;
		fps = (frame_id + 1.) / (total_frame_cost);
		cout << "this frame cost " << 1000 * frame_cost << " ms" << endl;
		total_frame_cost += frame_cost;
		before = after;
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh){
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
