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
#define FREQ 10

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image ipl_to_image(IplImage* src);
image get_image_from_stream_demo(CvCapture *cap, IplImage **cur_frame, int keyframe);
vector<Rect2d> process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static IplImage *cur_frame;
static Mat cur_frame_mat;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;
static int frame_id;
BBOX_tracker bbox_tracker;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void *fetch_in_thread(void *ptr)
{
	if(!cur_frame)
		cvReleaseImage(&cur_frame);
	double before = get_wall_time();
	cur_frame = cvQueryFrame(cap);
	cur_frame_mat = cvarrToMat(cur_frame);
	CvRect roi = cvRect(800, 500, 416, 416);
	cvSetImageROI(cur_frame, roi);
	IplImage *cur_frame_crop = cvCreateImage(cvGetSize(cur_frame), cur_frame->depth, cur_frame->nChannels);
	cvCopy(cur_frame, cur_frame_crop, NULL);
	cvResetImageROI(cur_frame);
	double after = get_wall_time();
	cout << "preprocess cost " << 1000 * (after - before) << " ms" << endl;
	//  keyframe
	if(frame_id % FREQ == 0){
		image im;
		if (!cur_frame) im = make_empty_image(0, 0, 0);
		im = ipl_to_image(cur_frame_crop);
		rgbgr_image(im);
		in_s = im;
	}
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 1, 0, demo_hier_thresh);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("Objects:\n\n");

	vector<Rect2d> bboxes;
    bboxes = process_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	for(unsigned int i = 0; i < bboxes.size(); i++)
		rectangle(cur_frame_mat, bboxes[i], Scalar(0, 255, 0), 2, 1);
	rectangle(cur_frame_mat, Point(800, 500), Point(1216, 916), Scalar(0, 255, 0), 2, 1);
	imshow("demo", cur_frame_mat);
	waitKey(1);

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
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    int count = 0;
	namedWindow("demo", WINDOW_NORMAL);
	resizeWindow("demo", 640, 480);
	
    double before = get_wall_time();
    while(1){
        if(0){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
                show_image(disp, "Demo");
                int c = cvWaitKey(1);
                if (c == 10){
                    if(frame_skip == 0) frame_skip = 60;
                    else if(frame_skip == 4) frame_skip = 0;
                    else if(frame_skip == 60) frame_skip = 4;   
                    else frame_skip = 0;
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0){
                free_image(disp);
                disp  = det;
            }
            det   = in;
            det_s = in_s;
        }else{
            fetch_in_thread(0);
			//  keyframe
			if(frame_id % FREQ == 0){
				/* det   = in; */
				det_s = in_s;
				bbox_tracker = BBOX_tracker();
				detect_in_thread(0);
				/* disp = det; */
				/* free_image(disp); */
			}
			// non-keyframe
			else{
				bbox_tracker.SetFrame(cur_frame_mat);
				bbox_tracker.update();
				bbox_tracker.draw_tracking();
			}
			frame_id += 1;
        }
		double after = get_wall_time();
		double frame_cost = after - before;
		total_frame_cost += frame_cost;
		fps = (frame_id + 1.) / total_frame_cost;
		printf("\033[2J");
		printf("\033[1;1H");
		printf("\nFPS:%.1f\n", fps);
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
        int class1 = max_index(probs[i], classes);
        float prob = probs[i][class1];
        if(prob > thresh){
            int width = im.h * .012;

            if(0){
                width = pow(prob, 1.f/2.f)*10+1;
                alphabet = 0;
            }

            printf("%s: %.0f%%\n", names[class1], prob*100);
            int offset = class1*123457 % classes;

            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            int left  = (b.x - b.w / 2.) * 416;
            int right = (b.x + b.w / 2.) * 416;
            int top   = (b.y - b.h / 2.) * 416;
            int bot   = (b.y + b.h / 2.) * 416;
			left += 800;
			right += 800;
			top += 500;
			bot += 500;
			Rect bbox;
			bbox.x = left;
			bbox.y = top;
			bbox.width = right - left;
			bbox.height = bot - top;
			
			tmp_bbox_list.push_back(bbox);

            /* draw_box_width(im, left, top, right, bot, width, red, green, blue); */
            /* if (alphabet) { */
                /* image label = get_label(alphabet, names[class1], (im.h*.03)/10); */
                /* draw_label(im, top + width, left, label, rgb); */
            /* } */
        }
    }
	bbox_tracker.CleanObjects();
	bbox_tracker.SetObjects(tmp_bbox_list);
	bbox_tracker.SetFrame(cur_frame_mat);
	bbox_tracker.InitTracker();
	return tmp_bbox_list;
}
