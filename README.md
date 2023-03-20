# robustpointclouds

## Install stuff

- [Install Visual Studio Build Tools](https://stackoverflow.com/a/70007787)
- [wget for Windows](https://eternallybored.org/misc/wget/)
- `conda env create -f environment.yml`

## Prepare the KITTI dataset

- `prep.bat`

## Run perturbation on a detection model

This is an example carried out on 3dSSD detection model.

Dowload the checkpoint from: `https://github.com/open-mmlab/mmdetection3d/tree/master/configs/3dssd` and save it in the folder `robust-point-clouds\checkpoints`.

Create a new file in the `configs\3dssd` folder called `3dssd_4x4_kitti-3d-car-adversarial` based on the original file `3dssd_4x4_kitti-3d-car`. This is the config file.

Modify the class in the `3dssd_4x4_kitti-3d-car-adversarial` file to apply perturbation. I added lines 101 and 102 in the file to add an adversary.

To train the adversarial model, modify the `train.bat` file to give the path to the config file and checkpoint file; then run the bat file. 

After the training has ended suceessfully, the results will be logged in `lightning_logs\version_x` folder. To evaluate the loss, modify the `evaluate_loss.bat` file and run it. Then modify the `visualize_loss.bat` file and run to generate two image files called `loss.png` and  `perturbation.png` in the `lightning_logs\version_x` folder.
