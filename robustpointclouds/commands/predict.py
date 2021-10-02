from robustpointclouds.lightningmodule import mmdetection3dLightningModule
from robustpointclouds.datamodule import mmdetection3dDataModule
import os
import yaml
import copy
from robustpointclouds.p_tqdm import t_imap as mapper
from functools import partial
from mmdet3d.apis import show_result_meshlab
import torch
from mmdet3d.core.visualizer.open3d_vis import Visualizer


def evaluate_sample(sample, baseline, adversarial):
    _, sample = sample
    filepath = os.path.abspath(sample['img_metas'].data[0][0]['pts_filename'])
    filename = os.path.splitext(os.path.basename(filepath))[0]
    baseline_result, baseline_data = baseline.predict_file(filepath)
    adversarial_result, adversarial_data = adversarial.predict_file(filepath)

    points = sample["points"].data[0]

    for idx_i, i in enumerate(points):
        points[idx_i] = i.clone().to(adversarial.device)

    with torch.no_grad():
        voxels, num_points, coors = adversarial.model.voxelize(points)
        voxel_features = adversarial.model.voxel_encoder(
            voxels, num_points, coors)
        perturbation = adversarial.model.adversary(voxel_features)
        perturbed_features = voxel_features + perturbation

        voxels, num_points, coors = baseline.model.voxelize(points)
        voxel_features = baseline.model.voxel_encoder(voxels, num_points, coors)

    result = {
        "filename": filename,
        "baseline_result": baseline_result,
        "baseline_data": baseline_data,
        "adversarial_result": adversarial_result,
        "adversarial_data": adversarial_data,
        "voxel_features": voxel_features,
        "perturbation": perturbation,
        "perturbed_features": perturbed_features
    }

    return result


def show_features(features, outdir):
    vis = Visualizer(features, mode='xyz')
    show_path = os.path.join(outdir, 'features.png')
    vis.show(show_path)


def predict(config_file: str, baseline_config_file: str, checkpoint_file: str,
            lightning_outdir: str):
    lightning_outdir = os.path.abspath(lightning_outdir)
    lightning_config_file = os.path.join(lightning_outdir, "config.yaml")
    lightning_hparams_file = os.path.join(lightning_outdir, "hparams.yaml")
    with open(lightning_config_file, 'r') as file:
        lightning_config = yaml.safe_load(file)
    with open(lightning_hparams_file, 'r') as file:
        lightning_hparams = yaml.safe_load(file)

    baseline_hparams = copy.deepcopy(lightning_hparams)
    baseline_hparams["config_file"] = baseline_config_file

    data_module = mmdetection3dDataModule(config_file=config_file)
    baseline = mmdetection3dLightningModule(**baseline_hparams)
    adversarial = mmdetection3dLightningModule(**lightning_hparams)

    checkpoint = os.path.join(lightning_outdir, "checkpoints", "last.ckpt")
    print("Loading checkpoint: {}".format(checkpoint))

    adversarial = adversarial.load_from_checkpoint(checkpoint)

    baseline = baseline.eval().to("cuda")
    adversarial = adversarial.eval().to("cuda")

    data_loader = data_module.test_dataloader()

    predictor = partial(evaluate_sample,
                        baseline=baseline,
                        adversarial=adversarial)

    baseline_dir = os.path.join(lightning_outdir, "predictions", "baseline")
    adversarial_dir = os.path.join(lightning_outdir, "predictions",
                                   "adversarial")

    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(adversarial_dir, exist_ok=True)

    for result in mapper(predictor,
                         enumerate(data_loader),
                         total=len(data_loader)):

        baseline_result_path = os.path.join(baseline_dir, result["filename"])
        adversarial_result_path = os.path.join(adversarial_dir,
                                               result["filename"])

        show_result_meshlab(result["baseline_data"],
                            result["baseline_result"],
                            baseline_dir,
                            0.3,
                            show=True,
                            snapshot=True,
                            task='det')

        show_result_meshlab(result["adversarial_data"],
                            result["adversarial_result"],
                            adversarial_dir,
                            0.3,
                            show=True,
                            snapshot=True,
                            task='det')

        show_features(result["voxel_features"], baseline_result_path)
        show_features(result["perturbed_features"], adversarial_result_path)