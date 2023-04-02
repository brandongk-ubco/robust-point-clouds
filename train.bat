python train.py ^
    fit ^
    --trainer.max_epochs 10 ^
    --data.batch_size 12 ^
    --data.num_workers 12 ^
    --model.config_file "configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_adversarial.py" ^
    --model.checkpoint_file "checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20200620_230755-f2a38b9a.pth"