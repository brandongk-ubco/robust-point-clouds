from tools.create_data import kitti_data_prep

root_path = "./data/kitti"
info_prefix = "kitti"
version = "v1.0"
out_dir = "./data/kitti"

kitti_data_prep(root_path, info_prefix, version, out_dir)
