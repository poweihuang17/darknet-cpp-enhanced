#ifndef RTREE
#define RTREE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <assert.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

typedef struct {
	int lx, ly, rx, ry;
	int cx, cy;
	int camera_id;
	int hilbert_value;
	float prob;
	vector<int> child_index;
	vector<int> grid_index;
	int level;
} rtree_box;


void rot(int n, int *x, int *y, int rx, int ry);
int get_hilbert_value(int cx, int cy);

void initTree(int GRID_SIZE);
void Rtree_solution(Mat cur_frame, vector< vector<rtree_box> > cam, vector< vector<rtree_box> > cam_backup, int RTREE_BRANCH, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y);
void Grid_solution(Mat cur_frame, vector< vector<rtree_box> > cam, int GRID_SIZE, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y);
void Origin_solution(Mat cur_frame, vector< vector<rtree_box> > cam, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y);

#endif
