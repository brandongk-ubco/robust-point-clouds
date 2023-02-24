rem Install wget from here: https://eternallybored.org/misc/wget/
mkdir "data/kitti"
mkdir "data/kitti/ImageSets"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip --no-check-certificate --content-disposition -O ./data/kitti/data_object_velodyne.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip --no-check-certificate --content-disposition -O ./data/kitti/data_object_image_2.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip --no-check-certificate --content-disposition -O ./data/kitti/data_object_label_2.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip --no-check-certificate --content-disposition -O ./data/kitti/data_object_calib.zip
unzip -n ./data/kitti/data_object_velodyne.zip -d ./data/kitti
unzip -n ./data/kitti/data_object_image_2.zip -d ./data/kitti
unzip -n ./data/kitti/data_object_label_2.zip -d ./data/kitti
unzip -n ./data/kitti/data_object_calib.zip -d ./data/kitti
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt

mkdir "checkpoints"
wget -c https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --no-check-certificate --content-disposition -O checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth

python mmdetection3d/tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti