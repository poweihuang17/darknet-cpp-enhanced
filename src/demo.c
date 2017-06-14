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
#include <fstream>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>
#include <mutex>
#endif

#define FREQ 1
int NET_DIM = 416;
int mid_xoffset = 700;
int mid_yoffset = 400;

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float **probs;
static box *boxes;
static network net;
static Mat pre_frame;
static Mat cur_frame;
static Mat roi_mid_mat;
static Mat roi_right_mat;
static Mat roi_left_mat;
static image roi_mid_img;
static image roi_right_img;
static image roi_left_img;
static image cur_frame_img;
static image cur_frame_img_s;

static VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;
static int frame_id;
mutex mtx;
int online_training = 0;
int writepred = 1;
BBOX_tracker mid_tracker;
BBOX_tracker right_tracker;
BBOX_tracker left_tracker;
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
vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
vector<yolo_rect> process_detections_ot(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);

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
	
	if(online_training){
		cur_frame_img = mat_to_image(cur_frame);
		cur_frame_img_s = resize_image(cur_frame_img, NET_DIM, NET_DIM);
	}
	else{
		roi_mid_mat = cur_frame(Rect(mid_xoffset, mid_yoffset, NET_DIM, NET_DIM));
		roi_right_mat = cur_frame(Rect(mid_xoffset + NET_DIM, mid_yoffset, NET_DIM, NET_DIM));
		roi_left_mat = cur_frame(Rect(mid_xoffset - NET_DIM, mid_yoffset, NET_DIM, NET_DIM));
		//  keyframe
		if((frame_id) % FREQ == 0){
			roi_mid_img = mat_to_image(roi_mid_mat);
		}
		if((frame_id+1) % FREQ == 0){
			roi_right_img = mat_to_image(roi_right_mat);
		}
		if((frame_id+2) % FREQ == 0){
			roi_left_img = mat_to_image(roi_left_mat);
		}
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
vector<yolo_rect> detect_roi_ot(image roi){
	float nms = .2;
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
	vector<yolo_rect> bboxes = process_detections_ot(cur_frame_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	return bboxes;
}

void *detect_roi_in_thread(void *ptr){
	det_thread_arg arg = *(det_thread_arg *)ptr;
	vector<Rect2d> bbox_draw;
	vector<yolo_rect> bbox_ot;
	Scalar color;

	if((frame_id+arg.frame_offset) % FREQ == 0){
		//----- keyframe-----//
		//  detect roi
		if(online_training)
			bbox_ot = detect_roi_ot(arg.roi_img);
		else
			bbox_draw = detect_roi(arg.roi_img);

		free_image(arg.roi_img);
		//  initial tracker
		*arg.tracker = BBOX_tracker();
		arg.tracker->SetObjects(bbox_draw);
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->InitTracker();
		color = Scalar(0, 255, 0);
	}
	else{
		//-----non-keyframe-----//
		//  track roi
		arg.tracker->SetROI(arg.roi_mat);
		arg.tracker->update();
		bbox_draw = arg.tracker->GetObjects();
		color = Scalar(0, 0, 255);
	}
	mtx.lock();
	//  write prediction
	if(writepred){
		FILE *online_label_file;
		char online_label_name[100] = {0};
		char online_label_data[100] = {0};
		sprintf(online_label_name, "./online_data/%s/predictions/%s_%d/%d.txt", video_file, weight_file, NET_DIM, frame_id);	
		online_label_file = fopen(online_label_name, "a");
		vector<Rect2d> bbox_wp = bbox_draw;
		for(unsigned int i = 0; i < bbox_wp.size(); i++){
			bbox_wp.at(i).x += arg.x_offset;
			bbox_wp.at(i).y += arg.y_offset;
			sprintf(online_label_data, "%d %f %f %f %f\n", 0, bbox_wp.at(i).x, bbox_wp.at(i).y, bbox_wp.at(i).width, bbox_wp.at(i).height);
			fwrite(online_label_data, 1, strlen(online_label_data), online_label_file);	
			memset(online_label_data, 0, sizeof(online_label_data));
		}
		fclose(online_label_file);
	}
	//  write image and label for online training
	if(online_training){
		FILE *online_label_file;
		char online_label_name[100] = {0};
		char online_image_name[100] = {0};
		char online_label_data[100] = {0};
		sprintf(online_label_name, "./online_data/%s/labels/%d.txt", video_file, frame_id);	
		sprintf(online_image_name, "./online_data/%s/images/%d.jpg", video_file, frame_id);	
		online_label_file = fopen(online_label_name, "w");
		for(unsigned int i = 0; i < bbox_ot.size(); i++){
			float x = bbox_ot.at(i).x;
			float y = bbox_ot.at(i).y;
			float w = bbox_ot.at(i).width;
			float h = bbox_ot.at(i).height;
			int left  = (x - w/2.) * 1920;
			int right = (x + w/2.) * 1920;
			int top   = (y - h/2.) * 1080;
			int bot   = (y + h/2.) * 1080;
			int width = right - left;
			Rect b(left, top, right-left, bot-top);
			/* if(bbox_ot.at(i).prob < 0.3){ */
				/* rectangle(cur_frame, b, Scalar(0, 0, 0), CV_FILLED, 1); */
			/* } */
			/* else{ */
			if(width < 500){
				/* rectangle(cur_frame, b, Scalar(0, 0, 255), 10, 1); */
				sprintf(online_label_data, "%d %f %f %f %f\n", 0, x, y, w, h);
				fwrite(online_label_data, 1, strlen(online_label_data), online_label_file);	
				memset(online_label_data, 0, sizeof(online_label_data));
			}
			/* } */
		}
		imwrite(online_image_name, cur_frame);
		fclose(online_label_file);
	}
	//  if not online training then draw boxes for show
	else{
		for(unsigned int i = 0; i < bbox_draw.size(); i++){
			bbox_draw.at(i).x += arg.x_offset;
			bbox_draw.at(i).y += arg.y_offset;
			rectangle(cur_frame, bbox_draw[i], color, 10, 1);
		}
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
    demo_thresh = thresh;
    demo_hier_thresh = hier_thresh;
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
	video_file = strdup(filename);
	video_file = strtok(basename(video_file), ".");
	weight_file = strtok(basename(weightfile), ".");
	char dir_name[100];
	sprintf(dir_name, "./online_data/%s/predictions/%s_%d/", video_file, weight_file, NET_DIM);
	mkpath(dir_name, S_IRWXU);
	char clean_cmd[100];
	sprintf(clean_cmd, "rm -rf %s/*", dir_name);
	system(clean_cmd);
	sprintf(dir_name, "./online_data/%s/labels/", video_file, weight_file);
	mkpath(dir_name, S_IRWXU);
	sprintf(dir_name, "./online_data/%s/images/", video_file, weight_file);
	mkpath(dir_name, S_IRWXU);
    layer l = net.layers[net.n - 1];
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));
	pthread_t detect_mid_thread, detect_right_thread, detect_left_thread;
	/* namedWindow("demo", WINDOW_NORMAL); */
	/* resizeWindow("demo", 1352, 1013); */
    double before = get_wall_time();
	int k = 0;
    while(1){
		printf("\033[2J");
		printf("\033[1;1H");
		printf("[frame id:%d]\n", frame_id);
		printf("FPS:%.1f\n", fps);
		fetch_in_thread(0);
		char w_f[100] = {0};
		if((frame_id+1) % 900 == 0){
			sprintf(w_f, "/home/newslab/yolo_weights/night_train%d.weights", k);
			cout << "change weight file: " << w_f << endl;
			putText(cur_frame, 
				"Download Model!",
				Point(5,0), // Coordinates
				FONT_HERSHEY_COMPLEX_SMALL, // Font
				1.0, // Scale. 2.0 = 2x bigger
				Scalar(255,255,255), // Color
				1, // Thickness
				CV_AA); // Anti-alias
			load_weights(&net, w_f);
			k += 1;
		}
		if(!online_training){
			//  create mid roi thread
			det_thread_arg mid_arg;
			mid_arg.roi_img = roi_mid_img;
			mid_arg.roi_mat = roi_mid_mat;
			mid_arg.tracker = &mid_tracker;
			mid_arg.x_offset = mid_xoffset;
			mid_arg.y_offset = mid_yoffset;
			mid_arg.frame_offset = 0;
			pthread_create(&detect_mid_thread, 0, detect_roi_in_thread, (void *)&mid_arg);
			pthread_join(detect_mid_thread, 0);
			//  create right roi thread
			det_thread_arg right_arg;
			right_arg.roi_img = roi_right_img;
			right_arg.roi_mat = roi_right_mat;
			right_arg.tracker = &right_tracker;
			right_arg.x_offset = mid_xoffset + NET_DIM;
			right_arg.y_offset = mid_yoffset;
			right_arg.frame_offset = 1;
			pthread_create(&detect_right_thread, 0, detect_roi_in_thread, (void *)&right_arg);
			pthread_join(detect_right_thread, 0);
			//  create left roi thread
			det_thread_arg left_arg;
			left_arg.roi_img = roi_left_img;
			left_arg.roi_mat = roi_left_mat;
			left_arg.tracker = &left_tracker;
			left_arg.x_offset = mid_xoffset - NET_DIM;
			left_arg.y_offset = mid_yoffset;
			left_arg.frame_offset = 2;
			pthread_create(&detect_left_thread, 0, detect_roi_in_thread, (void *)&left_arg);
			pthread_join(detect_left_thread, 0);
			//  waiting for thread completion
		}else{
			det_thread_arg mid_arg;
			mid_arg.roi_img = cur_frame_img_s;
			mid_arg.tracker = &mid_tracker;
			mid_arg.x_offset = 0;
			mid_arg.y_offset = 0;
			mid_arg.frame_offset = 0;
			pthread_create(&detect_mid_thread, 0, detect_roi_in_thread, (void *)&mid_arg);
			pthread_join(detect_mid_thread, 0);
			free_image(cur_frame_img);
		}
		rectangle(cur_frame, Point(mid_xoffset - NET_DIM, mid_yoffset), Point(mid_xoffset, mid_yoffset + NET_DIM), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame, Point(mid_xoffset, mid_yoffset), Point(mid_xoffset + NET_DIM, mid_yoffset + NET_DIM), Scalar(255, 255, 0), 2, 1);
		rectangle(cur_frame, Point(mid_xoffset + NET_DIM, mid_yoffset), Point(mid_xoffset + NET_DIM * 2, mid_yoffset + NET_DIM), Scalar(255, 255, 0), 2, 1);
		imshow("demo", cur_frame);
		waitKey(1);
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

vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes){
    int i;
	vector<Rect2d> tmp_bbox_list;
    for(i = 0; i < num; ++i){
		int class1 = max_index(probs[i], classes);
		if(class1 == 0){
			float prob = probs[i][class1];
			if(prob > thresh){
				printf("%s: %.0f%%\n", names[0], prob*100);
				box b = boxes[i];
				int left  = (b.x - b.w/2.) * im.w;
				int right = (b.x + b.w/2.) * im.w;
				int top   = (b.y - b.h/2.) * im.h;
				int bot   = (b.y + b.h/2.) * im.h;
				Rect bbox;
				bbox.x = left;
				bbox.y = top;
				bbox.width = right - left;
				bbox.height = bot - top;
				tmp_bbox_list.push_back(bbox);
			}
        }
    }
	return tmp_bbox_list;
}
vector<yolo_rect> process_detections_ot(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes){
    int i;
	vector<yolo_rect> tmp_bbox_list;
    for(i = 0; i < num; ++i){
		int class1 = max_index(probs[i], classes);
		if(class1 == 0){
			float prob = probs[i][class1];
			if(prob > thresh){
				printf("%s: %.0f%%\n", names[0], prob*100);
				box b = boxes[i];
				yolo_rect bbox;
				bbox.x = b.x;
				bbox.y = b.y;
				bbox.width = b.w;
				bbox.height = b.h;
				bbox.prob = prob;
				tmp_bbox_list.push_back(bbox);
			}
		}
    }
	return tmp_bbox_list;
}
