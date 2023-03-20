python train.py ^
    fit ^
    --trainer.max_epochs 10 ^
    --data.batch_size 12 ^
    --data.num_workers 12 ^
    --model.config_file "configs/3dssd/3dssd_4x4_kitti-3d-car-adversarial.py" ^
    --model.checkpoint_file "checkpoints/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth"