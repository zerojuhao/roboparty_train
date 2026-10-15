"""Evaluate saved estimators on identical raw observations, without physics rollout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
from sim2sim_rp1_pufo import Policy


def error_metrics(prediction, truth):
    if prediction.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 3 or not len(truth):
        raise ValueError("Expected nonempty matching [sample, 3] velocity arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite velocity input")
    error = prediction - truth
    stopped = np.linalg.norm(truth[:, :2], axis=1) < .1
    result = dict(samples=len(truth), mae_m_s=np.abs(error).mean(0).tolist(),
                  rmse_m_s=np.sqrt((error**2).mean(0)).tolist(), bias_m_s=error.mean(0).tolist(),
                  stopped_samples=int(stopped.sum()), stopped_fraction=float(stopped.mean()))
    result["stopped_mae_m_s"] = np.abs(error[stopped]).mean(0).tolist() if stopped.any() else None
    result["false_motion_when_stopped_fraction"] = (
        float((np.linalg.norm(prediction[stopped, :2], axis=1) > .2).mean()) if stopped.any() else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    corpora = []
    for path in args.corpus:
        with np.load(path) as data:
            obs = {key.removeprefix("obs__"): torch.from_numpy(data[key].copy())
                   for key in data.files if key.startswith("obs__")}
            truth = data["true_velocity"].copy()
        if any(len(value) != len(truth) for value in obs.values()):
            raise ValueError("Corpus observation sample counts differ")
        corpora.append((path, obs, truth))
    reports = []
    with torch.inference_mode():
        for checkpoint in args.checkpoint:
            policy = Policy(checkpoint)
            if not policy.actor.estimate_velocity:
                raise ValueError("Checkpoint has no velocity estimator")
            per_corpus, predictions, truths = [], [], []
            for path, obs, truth in corpora:
                norm = policy.normalizer.normalize_dict(obs)
                latent = policy.actor.depth_encoder(obs["depth_image"])
                state = torch.cat([norm[key] for key in ("state", "last_action", "history_actor")], -1)
                prediction = policy.actor.velocity_estimator(torch.cat([state, latent], -1)).numpy()
                predictions.append(prediction)
                truths.append(truth)
                per_corpus.append(dict(corpus=str(path.resolve()), metrics=error_metrics(prediction, truth)))
            with checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            reports.append(dict(checkpoint=str(checkpoint.resolve()), step=policy.step,
                                sha256=digest,
                                corpora=per_corpus,
                                combined=error_metrics(np.concatenate(predictions), np.concatenate(truths))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(scope="Identical raw observations and truth; each checkpoint uses its saved normalizer",
                  results=reports)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
