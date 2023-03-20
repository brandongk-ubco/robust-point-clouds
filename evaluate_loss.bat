python -m robustpointclouds ^
    evaluate-loss "configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car-adversarial.py" ^
    "configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py" ^
    "checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth" ^
    "lightning_logs/version_0"