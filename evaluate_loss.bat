python -m robustpointclouds ^
    evaluate-loss "configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_adversarial.py" ^
    "configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py" ^
    "checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20200620_230755-f2a38b9a.pth" ^
    "lightning_logs/version_11"