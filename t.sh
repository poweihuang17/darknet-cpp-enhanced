for((i=4;i<=12;++i))
do
	# ./darknet-cpp detector demo data/voc-kitti.data cfg/yolo-voc-kitti.cfg ../yolo_weights/yolov2_voc_kitti_rand.weights ./online_data/night_train$i/video/night_train$i.mp4
	./darknet-cpp detector demo data/voc-kitti.data cfg/tiny-yolo-voc-kitti.cfg ../yolo_weights/night_train$i.weights online_data/night_valid$i/video/night_valid$i.mp4
done
