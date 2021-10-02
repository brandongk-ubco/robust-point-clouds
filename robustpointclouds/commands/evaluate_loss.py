from robustpointclouds.lightningmodule import mmdetection3dLightningModule
from robustpointclouds.datamodule import mmdetection3dDataModule
import os
import yaml
import copy
import torch
import pandas as pd
from robustpointclouds.p_tqdm import t_imap as mapper
from functools import partial


def evaluate_sample(sample, baseline, adversarial):
    batch_idx, batch = sample
    baseline_loss = baseline.test_step(batch, batch_idx)
    adversarial_loss = adversarial.test_step(batch, batch_idx)

    adversarial_loss = {
        "adversarial_" + str(key): val[0] if type(val) is list else val
        for key, val in adversarial_loss.items()
    }
    baseline_loss = {
        "baseline_" + str(key): val[0] if type(val) is list else val
        for key, val in baseline_loss.items()
    }

    result = {**baseline_loss, **adversarial_loss}
    result = {
        key: val.detach().cpu().numpy().item() for key, val in result.items()
    }

    return result


def evaluate_loss(config_file: str, baseline_config_file: str,
                  checkpoint_file: str, lightning_outdir: str):
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
    data_loader = data_module.val_dataloader()
    results = []

    evaluator = partial(evaluate_sample,
                        baseline=baseline,
                        adversarial=adversarial)
    with torch.no_grad():
        for result in mapper(evaluator,
                             enumerate(data_loader),
                             total=len(data_loader)):
            results.append(result)

    results = pd.DataFrame(results)
    outfile = os.path.join(lightning_outdir, "evaluation_results.csv")
    results.to_csv(outfile)
