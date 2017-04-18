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
#define FREQ 1

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image get_image_from_stream_demo(CvCapture *cap, IplImage **cur_frame, int keyframe);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static IplImage *cur_frame;
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
	if(frame_id % FREQ == 0){
		double before = get_wall_time();
    	in = get_image_from_stream_demo(cap, &cur_frame, 1);
		double after = get_wall_time();
		cout << "get image cost " << after - before << " seconds" << endl;
    	if(!in.data){
        	error("Stream closed.");
    	}
		before = get_wall_time();
    	in_s = resize_image(in, net.w, net.h);
		after = get_wall_time();
		cout << "resize image cost " << after - before << " seconds" << endl;

	}
	else{
		get_image_from_stream_demo(cap, &cur_frame, 0);
	}
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
	double before = get_wall_time();
    float *prediction = network_predict(net, X);
	double after = get_wall_time();
	cout << "predict cost " << after - before << " seconds" << endl;

    /* memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float)); */
    /* mean_arrays(predictions, FRAMES, l.outputs, avg); */
    /* l.output = avg; */

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0, demo_hier_thresh);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    /* images[demo_index] = det; */
    /* det = images[(demo_index + FRAMES/2 + 1)%FRAMES]; */
    /* demo_index = (demo_index + 1)%FRAMES; */

    process_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

    return 0;
}

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
	/* if(!prefix){ */
		/* cvNamedWindow("Demo", CV_WINDOW_NORMAL);  */
		/* cvMoveWindow("Demo", 0, 0); */
		/* cvResizeWindow("Demo", 1352, 1013); */
	/* } */
	
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
			if(frame_id % FREQ == 0){
				det   = in;
				det_s = in_s;
				bbox_tracker = BBOX_tracker();
				detect_in_thread(0);
				disp = det;
				/* show_image(disp, "Demo"); */
				/* cvWaitKey(1); */
				free_image(disp);
			}
			else{
				bbox_tracker.SetFrame(cur_frame);
				bbox_tracker.update();
				bbox_tracker.draw_tracking();
			}
			frame_id += 1;
        }
		double after = get_wall_time();
		float curr = 1./(after - before);
		fps = curr;
		before = after;
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

void process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
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

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

			Rect bbox;
			bbox.x = left;
			bbox.y = top;
			bbox.width = right - left;
			bbox.height = bot - top;
			
			tmp_bbox_list.push_back(bbox);

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, names[class1], (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
            }
        }
    }
	bbox_tracker.CleanObjects();
	bbox_tracker.SetObjects(tmp_bbox_list);
	bbox_tracker.SetFrame(cur_frame);
	bbox_tracker.InitTracker();
}