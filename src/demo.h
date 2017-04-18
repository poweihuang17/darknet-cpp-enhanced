#ifndef DEMO
#define DEMO

#include "image.h"
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh);
void process_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);

#endif
