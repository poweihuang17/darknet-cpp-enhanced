#include "rtree.h"

vector< vector< vector<rtree_box> > > rtree;
vector< vector<rtree_box> > rtree_cam;
vector<rtree_box> rtree_root;

vector< vector<rtree_box> > grid;
vector< vector<rtree_box> > grid_cam;
vector<rtree_box> grid_root;

float overlap1_thresh = 0.5, overlap2_thresh = 0.25;

#define WEIGHT 70 //use 10~60 to define distance weight coefficient, 0 means normal

#if WEIGHT == 10
	float weight_left = 84.3/(84.3+65.2+15.2)*3;
	float weight_mid = 65.2/(84.3+65.2+15.2)*3;
	float weight_right = 15.2/(84.3+65.2+15.2)*3;
#elif WEIGHT == 20
	float weight_left = 86.9/(86.9+85.2+83.2)*3;
	float weight_mid = 85.2/(86.9+85.2+83.2)*3;
	float weight_right = 83.2/(86.9+85.2+83.2)*3;
#elif WEIGHT == 30
	float weight_left = 73.4/(73.4+86.9+85.4)*3;
	float weight_mid = 86.9/(73.4+86.9+85.4)*3;
	float weight_right = 85.4/(73.4+86.9+85.4)*3;
#elif WEIGHT == 40
	float weight_left = 52.5/(52.5+86.7+84.6)*3;
	float weight_mid = 86.7/(52.5+86.7+84.6)*3;
	float weight_right = 84.6/(52.5+86.7+84.6)*3;
#elif WEIGHT == 50
	float weight_left = 16.5/(16.5+70.6+84.8)*3;
	float weight_mid = 70.6/(16.5+70.6+84.8)*3;
	float weight_right = 84.8/(16.5+70.6+84.8)*3;
#elif WEIGHT == 60
	float weight_left = 4.5/(4.5+42.3+84.3)*3;
	float weight_mid = 42.3/(4.5+42.3+84.3)*3;
	float weight_right = 84.3/(4.5+42.3+84.3)*3;
#elif WEIGHT == 70
	float weight_left = 0/(0+13.1+65.7)*3;
	float weight_mid = 13.1/(0+13.1+65.7)*3;
	float weight_right = 65.7/(0+13.1+65.7)*3;
#elif WEIGHT == 0
	float weight_left = 1;
	float weight_mid = 1;
	float weight_right = 1;
#endif


bool my_cmp(const rtree_box &A, const rtree_box &B){
	return A.hilbert_value < B.hilbert_value;
}

void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

int get_hilbert_value(int cx, int cy){
	//cy = 512 - cy;
	int n = 9;
	int x, y, s, d=0;
	for (s=n/2; s>0; s/=2) {
		x = (cx & s) > 0;
		y = (cy & s) > 0;
		d += s * s * ((3 * x) ^ y);
		rot(s, &cx, &cy, x, y);
	}
	return d;
}

float grid_overlap1(rtree_box b1, rtree_box b2){
	int lx, ly, rx, ry;
	lx = max(b1.lx, b2.lx);
	ly = max(b1.ly, b2.ly);
	rx = min(b1.rx, b2.rx);
	ry = min(b1.ry, b2.ry);
	
	if(rx < lx) return 0;
	if(ry < ly) return 0;
	
	int intersection = (rx-lx+1)*(ry-ly+1);
	int b1_area = (b1.rx-b1.lx+1)*(b1.ry-b1.ly+1);
	int b2_area = (b2.rx-b2.lx+1)*(b2.ry-b2.ly+1);

	float area = (float)intersection / (float) (b1_area + b2_area - intersection);
	return area;
}
float rtree_overlap1(rtree_box b1, rtree_box b2){
	int lx, ly, rx, ry;
	lx = max(b1.lx, b2.lx);
	ly = max(b1.ly, b2.ly);
	rx = min(b1.rx, b2.rx);
	ry = min(b1.ry, b2.ry);
	
	if(rx < lx) return 0;
	if(ry < ly) return 0;
	
	int intersection = (rx-lx+1)*(ry-ly+1);
	int b1_area = (b1.rx-b1.lx+1)*(b1.ry-b1.ly+1);
	int b2_area = (b2.rx-b2.lx+1)*(b2.ry-b2.ly+1);

	float area = (float)intersection / (float) (b1_area + b2_area - intersection);
	return area;
}
float overlap1(rtree_box b1, rtree_box b2){
	int lx, ly, rx, ry;
	lx = max(b1.lx, b2.lx);
	ly = max(b1.ly, b2.ly);
	rx = min(b1.rx, b2.rx);
	ry = min(b1.ry, b2.ry);
	
	/*
	if(rx < lx) return 0;
	if(ry < ly) return 0;
	*/

	int intersection = (rx-lx+1)*(ry-ly+1);
	int b1_area = (b1.rx-b1.lx+1)*(b1.ry-b1.ly+1);
	int b2_area = (b2.rx-b2.lx+1)*(b2.ry-b2.ly+1);

	float area = (float)intersection / (float) (b1_area + b2_area - intersection);
	return area;
}

float overlap2(rtree_box b1, rtree_box b2, rtree_box b3){
	int lx, ly, rx, ry;
	lx = max(max(b1.lx, b2.lx), b3.lx);
	ly = max(max(b1.ly, b2.ly), b3.ly);
	rx = min(max(b1.rx, b2.rx), b3.rx);
	ry = min(max(b1.ry, b2.ry), b3.ry);
	
	int intersection = (rx-lx+1)*(ry-ly+1);
	
	lx = max(b1.lx, b2.lx);
	ly = max(b1.ly, b2.ly);
	rx = min(b1.rx, b2.rx);
	ry = min(b1.ry, b2.ry);
	int inter1_area = (rx-lx+1)*(ry-ly+1);
	
	lx = max(b2.lx, b3.lx);
	ly = max(b2.ly, b3.ly);
	rx = min(b2.rx, b3.rx);
	ry = min(b2.ry, b3.ry);
	int inter2_area = (rx-lx+1)*(ry-ly+1);
	
	lx = max(b1.lx, b3.lx);
	ly = max(b1.ly, b3.ly);
	rx = min(b1.rx, b3.rx);
	ry = min(b1.ry, b3.ry);
	int inter3_area = (rx-lx+1)*(ry-ly+1);

	int b1_area = (b1.rx-b1.lx+1)*(b1.ry-b1.ly+1);
	int b2_area = (b2.rx-b2.lx+1)*(b2.ry-b2.ly+1);
	int b3_area = (b3.rx-b3.lx+1)*(b3.ry-b3.ly+1);
	int union_area = b1_area + b2_area + b3_area - inter1_area - inter2_area - inter3_area + intersection;

	float area = (float)intersection / (float)union_area;
	return area;
}

void initTree(int GRID_SIZE){
	// init R-tree
	vector<rtree_box> list1;
	vector< vector<rtree_box> > list2;
	rtree.push_back(list2);  rtree.push_back(list2);  rtree.push_back(list2);
	rtree_cam.push_back(list1);  rtree_cam.push_back(list1);  rtree_cam.push_back(list1);
	
	// init Grid
	for(unsigned int x=0; x<GRID_SIZE*GRID_SIZE; x++){
		grid.push_back(list1);
	}
	grid_cam.push_back(list1);  grid_cam.push_back(list1);  grid_cam.push_back(list1);
}

void Construct_rtree(vector< vector<rtree_box> > cam, vector< vector<rtree_box> > cam_backup, int RTREE_BRANCH){
	unsigned int i, j;
	int count, num;
	vector<rtree_box> temp_list;
	
	for(i=0; i<3; i++){
		int level = 1;
		sort(cam_backup[i].begin(), cam_backup[i].end(), my_cmp);
		rtree[i].push_back(cam[i]);

		if(rtree[i][0].size() == 0){
			rtree_box temp2;
			temp2.lx = 0; temp2.ly = 0; temp2.rx = 0; temp2.ry = 0;
			temp2.level = 1;
			temp2.camera_id = i;
			rtree_root.push_back(temp2);
		}
		else if(rtree[i][0].size() == 1){
			rtree_box temp2;
			temp2.lx = rtree[i][0][0].lx; temp2.ly = rtree[i][0][0].ly; temp2.rx = rtree[i][0][0].rx; temp2.ry = rtree[i][0][0].ry;
			temp2.level = 1;
			temp2.child_index.push_back(0);
			temp2.camera_id = i;
			rtree_root.push_back(temp2);
		}
		else{
			while(rtree[i][level-1].size() > 1){
				count = 0;
				while(count < rtree[i][level-1].size()){
					if(count + RTREE_BRANCH <= rtree[i][level-1].size()){
						num = RTREE_BRANCH;
					}
					else{
						num = rtree[i][level-1].size() - count;
					}
					
					rtree_box temp;
					vector<rtree_box> b;
					
					for(j=0; j<num; j++){
						b.push_back(rtree[i][level-1][count]);
						temp.child_index.push_back(count);
						count++;
					}

					temp.lx=b[0].lx; temp.ly=b[0].ly; temp.rx=b[0].rx; temp.ry=b[0].ry;
					for(j=1; j<num; j++){
						temp.lx = min(temp.lx, b[j].lx);	
						temp.ly = min(temp.ly, b[j].ly);	
						temp.rx = max(temp.rx, b[j].rx);	
						temp.ry = max(temp.ry, b[j].ry);
					}
					temp.level = level;
					temp.camera_id = i;
					temp_list.push_back(temp);
				}
				rtree[i].push_back(temp_list);
				temp_list.clear();
				level++;
			}
			
			level--;
			rtree_box temp2;
			temp2.lx = rtree[i][level][0].lx; temp2.ly = rtree[i][level][0].ly; temp2.rx = rtree[i][level][0].rx; temp2.ry = rtree[i][level][0].ry;
			temp2.level = level;
			temp2.child_index = rtree[i][level][0].child_index;
			temp2.camera_id = i;
			rtree_root.push_back(temp2);
		}
	}
}

void Find_related_box(rtree_box target, rtree_box candidate){
	unsigned int i, j, k;

	if(rtree_overlap1(target, candidate) > 0.0001){
		if( candidate.level == 0 ){
			rtree_cam[candidate.camera_id].push_back(candidate);
		}
		else{
			//DFS search internal node
			for(k=0; k<candidate.child_index.size(); k++){
				Find_related_box(target, rtree[candidate.camera_id][candidate.level-1][candidate.child_index[k]]);
			}
		}
	}
}

void Rtree_solution(Mat cur_frame, vector< vector<rtree_box> > cam, vector< vector<rtree_box> > cam_backup, int RTREE_BRANCH, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y){
	// rtree solution
	unsigned int i, j, k;
	vector<Rect2d> demorgan_list;

	rtree.clear();
	rtree_cam.clear();
	rtree_root.clear();	

	Construct_rtree(cam, cam_backup, RTREE_BRANCH);


	for(i = 0; i<cam[0].size(); i++){
		rtree_cam[0].clear();  rtree_cam[1].clear();  rtree_cam[2].clear();
		Find_related_box(cam[0][i], rtree_root[1]);
		Find_related_box(cam[0][i], rtree_root[2]);
		
		
		for(j=0; j<rtree_cam[1].size(); j++){
			if(rtree_overlap1(cam[0][i], rtree_cam[1][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<rtree_cam[2].size(); k++){
					if(overlap2(cam[0][i], rtree_cam[1][j], rtree_cam[2][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam2, cam3
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-rtree_cam[1][j].prob), weight_mid) * pow((1-rtree_cam[2][k].prob), weight_right);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam2
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-rtree_cam[1][j].prob), weight_mid);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}

	for(i = 0; i<cam[0].size(); i++){
		rtree_cam[0].clear();  rtree_cam[1].clear();  rtree_cam[2].clear();
		Find_related_box(cam[0][i], rtree_root[1]);
		Find_related_box(cam[0][i], rtree_root[2]);
		for(j=0; j<rtree_cam[2].size(); j++){
			if(rtree_overlap1(cam[0][i], rtree_cam[2][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<rtree_cam[1].size(); k++){
					if(overlap2(cam[0][i], rtree_cam[2][j], rtree_cam[1][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam3, cam2
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-rtree_cam[2][j].prob), weight_right) * pow((1-rtree_cam[1][k].prob), weight_mid);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-rtree_cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}
	
	for(i=0; i<cam[1].size(); i++){
		rtree_cam[0].clear();  rtree_cam[1].clear();  rtree_cam[2].clear();
		Find_related_box(cam[1][i], rtree_root[0]);
		Find_related_box(cam[1][i], rtree_root[2]);
		int flag = 2;
		for(j=0; j<rtree_cam[2].size(); j++){
			if(rtree_overlap1(cam[1][i], rtree_cam[2][j]) > overlap1_thresh){
				flag--;
				int get_overlap2_flag = 0;
				for(k=0; k<rtree_cam[0].size(); k++){
					if(overlap2(cam[1][i], rtree_cam[2][j], rtree_cam[0][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
					}
				}

				if(get_overlap2_flag != 1){
					// Do fusion function of cam2, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[1][i].prob), weight_mid) * pow((1-rtree_cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[1][i].lx + ROI_SHIFT_X;
						bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[1][i].rx - cam[1][i].lx;
						bbox.height = cam[1][i].ry - cam[1][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
		for(j=0; j<rtree_cam[0].size(); j++){
			if(rtree_overlap1(cam[1][i], rtree_cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}

		if(flag == 2){
			if(cam[1][i].prob > demo_thresh){
				// Directly draw cam2	
				Rect bbox;
				bbox.x = cam[1][i].lx + ROI_SHIFT_X;
				bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[1][i].rx - cam[1][i].lx;
				bbox.height = cam[1][i].ry - cam[1][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}
	
	for(i=0; i<cam[2].size(); i++){
		rtree_cam[0].clear();  rtree_cam[1].clear();  rtree_cam[2].clear();
		Find_related_box(cam[2][i], rtree_root[0]);
		Find_related_box(cam[2][i], rtree_root[1]);
		int flag=2;
		for(j=0; j<rtree_cam[1].size(); j++){
			if(rtree_overlap1(cam[2][i], rtree_cam[1][j]) > overlap1_thresh){
				flag--;
			}
		}
		for(j=0; j<rtree_cam[0].size(); j++){
			if(rtree_overlap1(cam[2][i], rtree_cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}
		if(flag == 2){
			if(cam[2][i].prob > demo_thresh){
				// Directly draw cam3
				Rect bbox;
				bbox.x = cam[2][i].lx + ROI_SHIFT_X;
				bbox.y = cam[2][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[2][i].rx - cam[2][i].lx;
				bbox.height = cam[2][i].ry - cam[2][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}
	
	
	// draw the box that detected by demorgan
	for(unsigned int i = 0; i < demorgan_list.size(); i++){
		rectangle(cur_frame, demorgan_list[i], Scalar(100,0,0), 5, 1);
	}
	
}

void Construct_grid(vector< vector<rtree_box> > cam, int GRID_SIZE){
	unsigned int i, j, k;
	
	for(i=0; i<GRID_SIZE; i++){
		for(j=0; j<GRID_SIZE; j++){
			rtree_box temp;
			temp.lx = 416/GRID_SIZE * j;
			temp.rx = 416/GRID_SIZE * (j+1);
			temp.ly = 416/GRID_SIZE * i;
			temp.ry = 416/GRID_SIZE * (i+1);
			grid_root.push_back(temp);
		}
	}

	for(i=0; i<3; i++){
		for(j=0; j<cam[i].size(); j++){
			rtree_box temp;
			temp.lx = cam[i][j].lx; temp.ly = cam[i][j].ly;	temp.rx = cam[i][j].rx;	temp.ry = cam[i][j].ry;
			temp.hilbert_value = cam[i][j].hilbert_value;
			temp.prob = cam[i][j].prob;
			temp.level = cam[i][j].level;
			temp.camera_id = cam[i][j].camera_id;
			for(k=0; k<grid_root.size(); k++){
				if(grid_overlap1(cam[i][j], grid_root[k]) > 0.0001){
					temp.grid_index.push_back(k);
				}
			}
			for(k=0; k<temp.grid_index.size(); k++){
				grid[k].push_back(temp);
			}
		}
	}


}

void Find_grid_box(rtree_box target, int i, vector<int> related_list){
	unsigned int j, k, h;
	
	for(j=0; j<grid[i].size(); j++){
		if( target.camera_id != grid[i][j].camera_id && grid_overlap1(target, grid[i][j]) > 0.0001){			
			int cam_id = grid[i][j].camera_id;
			if(grid_cam[cam_id].size() == 0){
				grid_cam[cam_id].push_back(grid[i][j]);
			}
			else{
				int exist_flag = 0;
				for(k=0; k<grid[i][j].grid_index.size(); k++){
					if(k >= i) break;
					for(h=0; h<related_list.size(); h++){
						if(k == h){
							exist_flag = 1;
						}
					}
				}
				for(k=0; k<1000; k++){
				}

				/*
				int exist_flag = 0;
				for(k=0; k<grid_cam[cam_id].size(); k++){
					if (grid_overlap1(grid_cam[cam_id][k], grid[i][j]) > 0.9 && grid_cam[cam_id][k].hilbert_value == grid[i][j].hilbert_value){
						exist_flag = 1;
						break;
					}
				}
				*/
				if(exist_flag == 0){
					grid_cam[cam_id].push_back(grid[i][j]);
				}
			}
		}
	}
}

void Grid_solution(Mat cur_frame, vector< vector<rtree_box> > cam, int GRID_SIZE, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y){
	// grid solution
	unsigned int i, j, k;
	vector<Rect2d> demorgan_list;
	
	grid.clear();
	grid_cam.clear();
	grid_root.clear();	

	Construct_grid(cam, GRID_SIZE);


	for(i = 0; i<cam[0].size(); i++){
		grid_cam[0].clear(); grid_cam[1].clear(); grid_cam[2].clear();
		vector<int> related_list;
		for(j=0; j<grid_root.size(); j++){
			if(grid_overlap1(cam[0][i], grid_root[j]) > 0.0001){
				related_list.push_back(j);
			}
		}
		for(j=0; j<related_list.size(); j++){
			Find_grid_box(cam[0][i], j, related_list);
		}
		for(j=0; j<grid_cam[1].size(); j++){
			if(grid_overlap1(cam[0][i], grid_cam[1][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<grid_cam[2].size(); k++){
					if(overlap2(cam[0][i], grid_cam[1][j], grid_cam[2][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam2, cam3
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-grid_cam[1][j].prob), weight_mid) * pow((1-grid_cam[2][k].prob), weight_right);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam2
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-grid_cam[1][j].prob), weight_mid);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}

	for(i = 0; i<cam[0].size(); i++){
		grid_cam[0].clear(); grid_cam[1].clear(); grid_cam[2].clear();
		vector<int> related_list;
		for(j=0; j<grid_root.size(); j++){
			if(grid_overlap1(cam[0][i], grid_root[j]) > 0.0001){
				related_list.push_back(j);
			}
		}
		for(j=0; j<related_list.size(); j++){
			Find_grid_box(cam[0][i], j, related_list);
		}
		for(j=0; j<grid_cam[2].size(); j++){
			if(grid_overlap1(cam[0][i], grid_cam[2][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<grid_cam[1].size(); k++){
					if(overlap2(cam[0][i], grid_cam[2][j], grid_cam[1][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam3, cam2
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-grid_cam[2][j].prob), weight_right) * pow((1-grid_cam[1][k].prob), weight_mid);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-grid_cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}
	
	for(i=0; i<cam[1].size(); i++){
		grid_cam[0].clear(); grid_cam[1].clear(); grid_cam[2].clear();
		vector<int> related_list;
		for(j=0; j<grid_root.size(); j++){
			if(grid_overlap1(cam[1][i], grid_root[j]) > 0.0001){
				related_list.push_back(j);
			}
		}
		for(j=0; j<related_list.size(); j++){
			Find_grid_box(cam[1][i], j, related_list);
		}
		int flag = 2;
		for(j=0; j<grid_cam[2].size(); j++){
			if(grid_overlap1(cam[1][i], grid_cam[2][j]) > overlap1_thresh){
				flag--;
				int get_overlap2_flag = 0;
				for(k=0; k<grid_cam[0].size(); k++){
					if(overlap2(cam[1][i], grid_cam[2][j], grid_cam[0][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
					}
				}

				if(get_overlap2_flag != 1){
					// Do fusion function of cam2, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[1][i].prob), weight_mid) * pow((1-grid_cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[1][i].lx + ROI_SHIFT_X;
						bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[1][i].rx - cam[1][i].lx;
						bbox.height = cam[1][i].ry - cam[1][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
		for(j=0; j<grid_cam[0].size(); j++){
			if(grid_overlap1(cam[1][i], grid_cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}

		if(flag == 2){
			if(cam[1][i].prob > demo_thresh){
				// Directly draw cam2	
				Rect bbox;
				bbox.x = cam[1][i].lx + ROI_SHIFT_X;
				bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[1][i].rx - cam[1][i].lx;
				bbox.height = cam[1][i].ry - cam[1][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}
	
	for(i=0; i<cam[2].size(); i++){
		grid_cam[0].clear(); grid_cam[1].clear(); grid_cam[2].clear();
		vector<int> related_list;
		for(j=0; j<grid_root.size(); j++){
			if(grid_overlap1(cam[2][i], grid_root[j]) > 0.0001){
				related_list.push_back(j);
			}
		}
		for(j=0; j<related_list.size(); j++){
			Find_grid_box(cam[2][i], j, related_list);
		}
		int flag=2;
		for(j=0; j<grid_cam[1].size(); j++){
			if(grid_overlap1(cam[2][i], grid_cam[1][j]) > overlap1_thresh){
				flag--;
			}
		}
		for(j=0; j<grid_cam[0].size(); j++){
			if(grid_overlap1(cam[2][i], grid_cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}
		if(flag == 2){
			if(cam[2][i].prob > demo_thresh){
				// Directly draw cam3
				Rect bbox;
				bbox.x = cam[2][i].lx + ROI_SHIFT_X;
				bbox.y = cam[2][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[2][i].rx - cam[2][i].lx;
				bbox.height = cam[2][i].ry - cam[2][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}
	
	// draw the box that detected by demorgan
	for(unsigned int i = 0; i < demorgan_list.size(); i++){
		rectangle(cur_frame, demorgan_list[i], Scalar(100,0,0), 5, 1);
	}
}


void Origin_solution(Mat cur_frame, vector< vector<rtree_box> > cam, float demo_thresh, int ROI_SHIFT_X, int ROI_SHIFT_Y){
	// origin solution
	unsigned int i, j, k;
	vector<Rect2d> demorgan_list;

	for(i = 0; i<cam[0].size(); i++){
		for(j=0; j<cam[1].size(); j++){
			if(overlap1(cam[0][i], cam[1][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<cam[2].size(); k++){
					if(overlap2(cam[0][i], cam[1][j], cam[2][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam2, cam3
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-cam[1][j].prob), weight_mid) * pow((1-cam[2][k].prob), weight_right);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam2
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-cam[1][j].prob), weight_mid);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}
	for(i = 0; i<cam[0].size(); i++){
		for(j=0; j<cam[2].size(); j++){
			if(overlap1(cam[0][i], cam[2][j]) > overlap1_thresh){
				int get_overlap2_flag = 0;
				for(k=0; k<cam[1].size(); k++){
					if(overlap2(cam[0][i], cam[2][j], cam[1][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
						// Do fusion function of cam1, cam3, cam2
						float demorgan;
						demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-cam[2][j].prob), weight_right) * pow((1-cam[1][k].prob), weight_mid);
						if(demorgan > demo_thresh){
							Rect bbox;
							bbox.x = cam[0][i].lx + ROI_SHIFT_X;
							bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
							bbox.width = cam[0][i].rx - cam[0][i].lx;
							bbox.height = cam[0][i].ry - cam[0][i].ly;
							demorgan_list.push_back(bbox);
						}
					}
				}
				
				if(get_overlap2_flag != 1){
					// Do fusion function of cam1, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[0][i].prob), weight_left) * pow((1-cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[0][i].lx + ROI_SHIFT_X;
						bbox.y = cam[0][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[0][i].rx - cam[0][i].lx;
						bbox.height = cam[0][i].ry - cam[0][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
	}
	for(i=0; i<cam[1].size(); i++){
		int flag = 2;
		for(j=0; j<cam[2].size(); j++){
			if(overlap1(cam[1][i], cam[2][j]) > overlap1_thresh){
				flag--;
				int get_overlap2_flag = 0;
				for(k=0; k<cam[0].size(); k++){
					if(overlap2(cam[1][i], cam[2][j], cam[0][k]) > overlap2_thresh){
						get_overlap2_flag = 1;
					}
				}

				if(get_overlap2_flag != 1){
					// Do fusion function of cam2, cam3
					float demorgan;
					demorgan = 1 - pow((1-cam[1][i].prob), weight_mid) * pow((1-cam[2][j].prob), weight_right);
					if(demorgan > demo_thresh){
						Rect bbox;
						bbox.x = cam[1][i].lx + ROI_SHIFT_X;
						bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
						bbox.width = cam[1][i].rx - cam[1][i].lx;
						bbox.height = cam[1][i].ry - cam[1][i].ly;
						demorgan_list.push_back(bbox);
					}
				}
			}
		}
		for(j=0; j<cam[0].size(); j++){
			if(overlap1(cam[1][i], cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}

		if(flag == 2){
			if(cam[1][i].prob > demo_thresh){
				// Directly draw cam2	
				Rect bbox;
				bbox.x = cam[1][i].lx + ROI_SHIFT_X;
				bbox.y = cam[1][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[1][i].rx - cam[1][i].lx;
				bbox.height = cam[1][i].ry - cam[1][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}
	for(i=0; i<cam[2].size(); i++){
		int flag=2;
		for(j=0; j<cam[1].size(); j++){
			if(overlap1(cam[2][i], cam[1][j]) > overlap1_thresh){
				flag--;
			}
		}
		for(j=0; j<cam[0].size(); j++){
			if(overlap1(cam[2][i], cam[0][j]) > overlap1_thresh){
				flag--;
			}
		}
		if(flag == 2){
			if(cam[2][i].prob > demo_thresh){
				// Directly draw cam3
				Rect bbox;
				bbox.x = cam[2][i].lx + ROI_SHIFT_X;
				bbox.y = cam[2][i].ly + ROI_SHIFT_Y;
				bbox.width = cam[2][i].rx - cam[2][i].lx;
				bbox.height = cam[2][i].ry - cam[2][i].ly;
				demorgan_list.push_back(bbox);
			}
		}
	}

	
	// draw the box that detected by demorgan
	for(unsigned int i = 0; i < demorgan_list.size(); i++){
		rectangle(cur_frame, demorgan_list[i], Scalar(100,0,0), 5, 1);
	}
}

