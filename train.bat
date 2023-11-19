python train.py ^
    fit ^
    --trainer.max_epochs 10 ^
    --data.batch_size 12 ^
    --data.num_workers 12 ^
    --model.config_file "mmdetection3d/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py" ^
    --model.checkpoint_file "checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"