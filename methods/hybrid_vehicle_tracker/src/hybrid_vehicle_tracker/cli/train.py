from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hybrid_vehicle_tracker.association.graph import build_candidate_graph
from hybrid_vehicle_tracker.cli.common import load_yaml, require_device, set_seed
from hybrid_vehicle_tracker.config import AssociationConfig, ModelConfig
from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.data.synthetic import (
    SyntheticSettings,
    SyntheticVehicleDataset,
)
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel
from hybrid_vehicle_tracker.types import VehicleObservation


def _focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = target * probability + (1.0 - target) * (1.0 - probability)
    return ((1.0 - pt).pow(gamma) * cross_entropy).mean()


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(-2, -1)) + 1.0
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def _hough_target(output, track_params: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(output.logits)
    slopes = output.slopes
    intercepts = output.intercepts
    for batch_index in range(track_params.shape[0]):
        for slope, intercept, valid in track_params[batch_index]:
            if valid < 0.5:
                continue
            slope_index = int(torch.argmin(torch.abs(slopes - slope)).item())
            intercept_index = int(torch.argmin(torch.abs(intercepts - intercept)).item())
            for ds in (-1, 0, 1):
                for di in (-1, 0, 1):
                    si = slope_index + ds
                    ii = intercept_index + di
                    if 0 <= si < target.shape[1] and 0 <= ii < target.shape[2]:
                        target[batch_index, si, ii] = max(
                            float(target[batch_index, si, ii]),
                            1.0 if ds == 0 and di == 0 else 0.5,
                        )
    return target


def _hough_loss(output, target: torch.Tensor) -> torch.Tensor:
    """Sparse focal loss plus hard-negative ranking for Hough maxima."""
    loss = _focal_loss(output.logits, target)
    ranking_terms: list[torch.Tensor] = []
    for batch_index in range(target.shape[0]):
        logits = output.logits[batch_index].reshape(-1)
        labels = target[batch_index].reshape(-1)
        positive = logits[labels >= 0.99]
        negative = logits[labels < 0.25]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        hard_count = min(negative.numel(), max(32, 12 * positive.numel()))
        hard_negative = torch.topk(negative, k=hard_count).values
        ranking_terms.append(F.softplus(-positive).mean() + F.softplus(hard_negative).mean())
    if ranking_terms:
        loss = loss + 0.50 * torch.stack(ranking_terms).mean()
    return loss


def _perception_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    feature_rate_hz: float,
) -> torch.Tensor:
    centerline = batch["centerline"]
    crossing = batch["crossing"]
    foreground = centerline > 0.20
    center_loss = _focal_loss(outputs["centerline_logits"], centerline) + _dice_loss(
        outputs["centerline_logits"], centerline
    )
    if foreground.any():
        slow_loss = F.smooth_l1_loss(outputs["slowness"][foreground], batch["slowness"][foreground])
    else:
        slow_loss = outputs["slowness"].sum() * 0.0
    crossing_loss = _focal_loss(outputs["crossing_logits"], crossing)
    embedding_loss = _track_embedding_loss(
        outputs["embedding"], batch, feature_rate_hz=feature_rate_hz
    )
    return center_loss + 0.5 * slow_loss + 0.35 * crossing_loss + 0.20 * embedding_loss


def _track_embedding_loss(
    embedding: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    feature_rate_hz: float,
    margin: float = 0.65,
) -> torch.Tensor:
    """Pull observations from one vehicle together and separate crossing vehicles."""
    losses: list[torch.Tensor] = []
    time_bins = embedding.shape[-1]
    for batch_index in range(embedding.shape[0]):
        centroids: list[torch.Tensor] = []
        for track_index in range(batch["track_times"].shape[1]):
            observed = batch["track_observed"][batch_index, track_index] > 0.5
            channels = torch.where(observed)[0]
            if channels.numel() < 2:
                continue
            times = batch["track_times"][batch_index, track_index, channels]
            bins = torch.clamp(
                torch.round(times * feature_rate_hz).long(), 0, time_bins - 1
            )
            vectors = embedding[batch_index, :, channels, bins].transpose(0, 1)
            centroid = F.normalize(vectors.mean(dim=0), dim=0, eps=1e-6)
            centroids.append(centroid)
            losses.append((1.0 - vectors @ centroid).mean())
        if len(centroids) >= 2:
            matrix = torch.stack(centroids)
            similarity = matrix @ matrix.transpose(0, 1)
            upper = torch.triu_indices(len(centroids), len(centroids), offset=1, device=embedding.device)
            losses.append(F.relu(similarity[upper[0], upper[1]] - margin).mean())
    if not losses:
        return embedding.sum() * 0.0
    return torch.stack(losses).mean()


def _make_gnn_graph(
    scene: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    geometry,
    association_config: AssociationConfig,
    embedding_dim: int,
    window_s: float,
    rng: np.random.Generator,
):
    inputs = scene["input"].detach().cpu().numpy()
    times = scene["track_times"].detach().cpu().numpy()
    observed = scene["track_observed"].detach().cpu().numpy()
    network = torch.sigmoid(outputs["centerline_logits"])[0, 0].detach().cpu().numpy()
    crossing = torch.sigmoid(outputs["crossing_logits"])[0, 0].detach().cpu().numpy()
    embedding = outputs["embedding"][0].detach().cpu().numpy()
    feature_rate = inputs.shape[-1] / window_s
    rows: list[tuple[VehicleObservation, int]] = []
    for track_id in range(times.shape[0]):
        if not np.any(observed[track_id] > 0.5):
            continue
        for channel in np.flatnonzero(observed[track_id] > 0.5):
            time_s = float(times[track_id, channel])
            if not 0.0 <= time_s < window_s:
                continue
            time_bin = int(np.clip(round(time_s * feature_rate), 0, inputs.shape[-1] - 1))
            station = geometry.stations[int(channel)]
            rows.append(
                (
                    VehicleObservation(
                        observation_id=-1,
                        channel_index=int(channel),
                        station_id=station.station_id,
                        position_m=station.position_m,
                        time_s=time_s,
                        gauss_score=float(inputs[2, channel, time_bin]),
                        pre_score=float(1.0 - inputs[1, channel, time_bin]),
                        raw_energy=float(np.clip(0.5 + 0.5 * inputs[0, channel, time_bin], 0, 1)),
                        network_score=float(network[channel, time_bin]),
                        crossing_score=float(crossing[channel, time_bin]),
                        strong=True,
                        embedding=tuple(float(value) for value in embedding[:, channel, time_bin]),
                    ),
                    track_id,
                )
            )
    noise_count = int(rng.integers(30, 90))
    for _ in range(noise_count):
        channel = int(rng.integers(0, len(geometry)))
        time_bin = int(rng.integers(0, inputs.shape[-1]))
        station = geometry.stations[channel]
        rows.append(
            (
                VehicleObservation(
                    observation_id=-1,
                    channel_index=channel,
                    station_id=station.station_id,
                    position_m=station.position_m,
                    time_s=time_bin / feature_rate,
                    gauss_score=float(inputs[2, channel, time_bin]),
                    pre_score=float(1.0 - inputs[1, channel, time_bin]),
                    raw_energy=float(np.clip(0.5 + 0.5 * inputs[0, channel, time_bin], 0, 1)),
                    network_score=float(network[channel, time_bin]),
                    crossing_score=float(crossing[channel, time_bin]),
                    strong=False,
                    embedding=tuple(float(value) for value in embedding[:, channel, time_bin]),
                ),
                -1,
            )
        )
    rows.sort(key=lambda item: (item[0].position_m, item[0].time_s))
    observations = [row[0] for row in rows]
    track_ids = np.asarray([row[1] for row in rows], dtype=np.int64)
    for index, observation in enumerate(observations):
        observation.observation_id = index
    graph = build_candidate_graph(
        observations,
        geometry,
        association_config,
        duration_s=window_s,
        embedding_dim=embedding_dim,
    )
    if graph.edge_index.shape[1]:
        edge_labels = (
            (track_ids[graph.edge_index[0]] >= 0)
            & (track_ids[graph.edge_index[0]] == track_ids[graph.edge_index[1]])
        ).astype(np.float32)
    else:
        edge_labels = np.empty((0,), dtype=np.float32)
    merge_labels = np.asarray(
        [float(item.crossing_score >= 0.5 and track_id >= 0) for item, track_id in rows],
        dtype=np.float32,
    )
    return graph, edge_labels, merge_labels


def _update_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float = 0.995) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _configure_stage(model: HybridPerceptionModel, stage: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {
        "perception": [model.perception],
        "hough": [model.perception, model.hough],
        "gnn": [model.edge_gnn],
        "joint": [model.perception, model.hough, model.edge_gnn],
    }[stage]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hybrid vehicle tracker on CUDA")
    parser.add_argument("--config", default="configs/synthetic_training.yaml")
    parser.add_argument("--stage", choices=["perception", "hough", "gnn", "joint", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_yaml(args.config)
    seed = int(config.get("seed", 20260724))
    set_seed(seed)
    device = require_device(str(config.get("device", "cuda")))
    geometry = load_station_geometry(config["geometry_mapping"])
    window_s = float(config.get("window_s", 48.0))
    feature_rate = float(config.get("feature_rate_hz", 20.0))
    samples = int(args.samples or config.get("samples_per_epoch", 64))
    epochs = int(args.epochs or config.get("epochs_per_stage", 2))
    forbidden_keys = {"background_root", "background_days", "windows_per_day"} & set(config)
    if forbidden_keys:
        raise ValueError(
            "real-data background options are prohibited; remove: "
            + ", ".join(sorted(forbidden_keys))
        )
    dataset = SyntheticVehicleDataset(
        geometry,
        SyntheticSettings(
            simulator_version=str(config.get("simulator_version", "parametric_day11_morphology_v2")),
            window_s=window_s,
            feature_rate_hz=feature_rate,
            waveform_rate_hz=float(config.get("waveform_rate_hz", 200.0)),
            max_vehicles=int(config.get("max_vehicles", 20)),
            samples=samples,
            seed=seed,
            min_speed_kmh=float(config.get("min_speed_kmh", 60.0)),
            max_speed_kmh=float(config.get("max_speed_kmh", 90.0)),
            vehicle_rate=float(config.get("vehicle_rate", 5.0)),
            interaction_probability=float(config.get("interaction_probability", 0.55)),
            motion_direction=int(config.get("motion_direction", -1)),
            peakset_vehicle_min=int(config.get("peakset_vehicle_min", 5)),
            peakset_vehicle_max=int(config.get("peakset_vehicle_max", 8)),
            peakset_min_visible_channels=int(config.get("peakset_min_visible_channels", 5)),
            peakset_dead_channels=str(
                config.get("peakset_dead_channels", "5,6,15,16,22,36,38,45")
            ),
            peakset_false_event_min=int(config.get("peakset_false_event_min", 0)),
            peakset_false_event_max=int(config.get("peakset_false_event_max", 4)),
            peakset_missing_ratio_min=float(config.get("peakset_missing_ratio_min", 0.0)),
            peakset_missing_ratio_max=float(config.get("peakset_missing_ratio_max", 0.03)),
            peakset_gap_probability=float(config.get("peakset_gap_probability", 0.45)),
            peakset_gap_min_channels=int(config.get("peakset_gap_min_channels", 1)),
            peakset_gap_max_channels=int(config.get("peakset_gap_max_channels", 3)),
            peakset_background_noise=float(config.get("peakset_background_noise", 0.012)),
            peakset_vehicle_amp_min=float(config.get("peakset_vehicle_amp_min", 0.45)),
            peakset_vehicle_amp_max=float(config.get("peakset_vehicle_amp_max", 0.65)),
            peakset_boundary_vehicle_ratio=float(config.get("peakset_boundary_vehicle_ratio", 0.35)),
            peakset_boundary_time_margin_s=float(config.get("peakset_boundary_time_margin_s", 3.5)),
            peakset_isolated_fraction_min=float(config.get("peakset_isolated_fraction_min", 0.28)),
            peakset_isolated_fraction_max=float(config.get("peakset_isolated_fraction_max", 0.40)),
            peakset_outage_probability=float(config.get("peakset_outage_probability", 0.70)),
            peakset_outage_min_channels=int(config.get("peakset_outage_min_channels", 1)),
            peakset_outage_max_channels=int(config.get("peakset_outage_max_channels", 3)),
            peakset_outage_min_duration_s=float(config.get("peakset_outage_min_duration_s", 0.8)),
            peakset_outage_max_duration_s=float(config.get("peakset_outage_max_duration_s", 5.5)),
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config.get("batch_size", 2)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    model_config = ModelConfig(
        base_channels=int(config.get("base_channels", 12)),
        embedding_dim=int(config.get("embedding_dim", 32)),
        hough_slopes=int(config.get("hough_slopes", 17)),
        hough_intercept_step_s=float(config.get("hough_intercept_step_s", 0.5)),
        motion_direction=int(config.get("motion_direction", -1)),
    )
    model = HybridPerceptionModel(model_config).to(device)
    if args.resume:
        model.load_checkpoint(args.resume, map_location=str(device))
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    association_config = AssociationConfig(
        min_observations=3,
        min_span_m=200.0,
        motion_direction=int(config.get("motion_direction", -1)),
    )
    stages = ["perception", "hough", "gnn", "joint"] if args.stage == "all" else [args.stage]
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train_history.jsonl"
    if args.resume is None and args.stage == "all":
        log_path.write_text("", encoding="utf-8")
    positions = torch.from_numpy(geometry.positions_m.astype(np.float32)).to(device)
    rng = np.random.default_rng(seed)

    global_epoch = 0
    for stage in stages:
        model.train()
        if stage == "joint":
            teacher.load_state_dict(model.state_dict())
        trainable_parameters = _configure_stage(model, stage)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(config.get("learning_rate", 1e-3)),
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=2000)
        for epoch in range(epochs):
            # Resample the fully synthetic scene population.  Without this,
            # index-based deterministic seeds would replay the same small set
            # of scenes at every epoch.
            dataset.set_epoch(global_epoch)
            global_epoch += 1
            epoch_loss = 0.0
            batches = 0
            for raw_batch in loader:
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(stage != "gnn"):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(batch["input"])
                loss = torch.zeros((), device=device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if stage in {"perception", "hough", "joint"}:
                        loss = loss + _perception_loss(
                            outputs, batch, feature_rate_hz=feature_rate
                        )
                    if stage in {"hough", "joint"}:
                        hough = model.hough(
                            outputs["feature_map"],
                            positions,
                            duration_s=window_s,
                            feature_rate_hz=feature_rate,
                            evidence_map=torch.sigmoid(outputs["centerline_logits"]),
                        )
                        target = _hough_target(hough, batch["track_params"])
                        loss = loss + 0.5 * _hough_loss(hough, target)
                    if stage == "joint":
                        with torch.no_grad():
                            teacher_output = teacher(batch["input"])["centerline_logits"]
                        perturbed = batch["input"] + 0.02 * torch.randn_like(batch["input"])
                        student_augmented = model(perturbed)["centerline_logits"]
                        loss = loss + 0.10 * F.mse_loss(
                            torch.sigmoid(student_augmented), torch.sigmoid(teacher_output)
                        )
                if stage in {"gnn", "joint"}:
                    gnn_losses = []
                    for item_index in range(batch["input"].shape[0]):
                        scene = {key: value[item_index] for key, value in batch.items()}
                        scene_output = {key: value[item_index : item_index + 1] for key, value in outputs.items()}
                        graph, edge_labels, merge_labels = _make_gnn_graph(
                            scene,
                            scene_output,
                            geometry,
                            association_config,
                            model_config.embedding_dim,
                            window_s,
                            rng,
                        )
                        if not graph.edge_index.shape[1]:
                            continue
                        node_tensor = torch.from_numpy(graph.node_features).to(device)
                        edge_index = torch.from_numpy(graph.edge_index).to(device)
                        edge_features = torch.from_numpy(graph.edge_features).to(device)
                        edge_target = torch.from_numpy(edge_labels).to(device)
                        merge_target = torch.from_numpy(merge_labels).to(device)
                        edge_logits, merge_logits = model.edge_gnn(
                            node_tensor, edge_index, edge_features
                        )
                        positives = max(float(edge_target.sum().item()), 1.0)
                        negatives = max(float(edge_target.numel() - edge_target.sum().item()), 1.0)
                        pos_weight = torch.tensor(min(negatives / positives, 20.0), device=device)
                        gnn_losses.append(
                            F.binary_cross_entropy_with_logits(
                                edge_logits, edge_target, pos_weight=pos_weight
                            )
                            + 0.20 * F.binary_cross_entropy_with_logits(merge_logits, merge_target)
                        )
                    if gnn_losses:
                        loss = loss + torch.stack(gnn_losses).mean()
                if not loss.requires_grad:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss in stage={stage}, epoch={epoch + 1}, batch={batches + 1}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=5.0, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                if stage == "joint":
                    _update_ema(teacher, model)
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            record = {
                "stage": stage,
                "epoch": epoch + 1,
                "loss": epoch_loss / max(batches, 1),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "max_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))
        model.save_checkpoint(
            checkpoint_dir / f"hybrid_{stage}.pt",
            stage=stage,
            seed=seed,
            training_data=f"fully_synthetic_{dataset.settings.simulator_version}",
            simulator_version=dataset.settings.simulator_version,
            real_training_days=[],
            waveform_rate_hz=float(config.get("waveform_rate_hz", 200.0)),
            interaction_probability=float(config.get("interaction_probability", 0.55)),
            day11_waveform_statistics_used=dataset.settings.simulator_version
            in {"vehicle_peakset_complex_v2", "vehicle_peakset_complex", "realshape_complex_v2"},
            simulator_calibration_note=(
                "DAY11[:120000] Gauss peak statistics only; no real samples loaded"
                if dataset.settings.simulator_version
                in {"vehicle_peakset_complex_v2", "vehicle_peakset_complex", "realshape_complex_v2"}
                else "clean vehicle-peak-set waveform profile; no real samples loaded"
            ),
            day11_vehicle_identity_labels_used=False,
            day11_samples_used_for_gradient_training=False,
        )
    model.save_checkpoint(
        checkpoint_dir / "hybrid_final.pt",
        stages=stages,
        seed=seed,
        training_data=f"fully_synthetic_{dataset.settings.simulator_version}",
        simulator_version=dataset.settings.simulator_version,
        real_training_days=[],
        waveform_rate_hz=float(config.get("waveform_rate_hz", 200.0)),
        interaction_probability=float(config.get("interaction_probability", 0.55)),
        day11_waveform_statistics_used=dataset.settings.simulator_version
        in {"vehicle_peakset_complex_v2", "vehicle_peakset_complex", "realshape_complex_v2"},
        simulator_calibration_note=(
            "DAY11[:120000] Gauss peak statistics only; no real samples loaded"
            if dataset.settings.simulator_version
            in {"vehicle_peakset_complex_v2", "vehicle_peakset_complex", "realshape_complex_v2"}
            else (
                "clean vehicle-peak-set waveform profile; no real samples loaded"
                if dataset.settings.simulator_version.startswith("vehicle_peakset")
                else "hand-specified DAY11 morphology ranges; no real samples loaded"
            )
        ),
        day11_vehicle_identity_labels_used=False,
        day11_samples_used_for_gradient_training=False,
    )


if __name__ == "__main__":
    main()
