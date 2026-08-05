from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from repo_runtime import configure_repo_runtime

configure_repo_runtime(Path(__file__).resolve().parents[1])

import ultralytics.utils.loss as loss_mod
import ultralytics.utils.metrics as metrics_mod
import ultralytics.utils.tal as tal_mod

from .context import get_experiment_context
from .modules import DWTDown, DefectMorphGate, ECABlock, ResidualWTConv, SPDConv, WTConv, WTRefine, WeightedConcat

_PATCHED = False


def _nwd_components(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    c1 = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    c2 = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    wh1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(1e-6)
    wh2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(1e-6)
    center_term = (c1 - c2).pow(2).sum(-1)
    size_term = ((wh1 - wh2).pow(2).sum(-1)) * 0.25
    return torch.sqrt((center_term + size_term).clamp_min(1e-9))


def bbox_nwd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor, constant: float = 12.8) -> torch.Tensor:
    distance = _nwd_components(boxes1, boxes2)
    return torch.exp(-distance / constant)


def focaler_iou_similarity(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    lower: float = 0.0,
    upper: float = 0.95,
) -> torch.Tensor:
    iou = metrics_mod.bbox_iou(boxes1, boxes2, xywh=False, CIoU=True)
    denom = max(upper - lower, 1e-6)
    return ((iou - lower) / denom).clamp_(0.0, 1.0)


class ContextAwareTaskAlignedAssigner:
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        overlaps = super().iou_calculation(gt_bboxes, pd_bboxes)
        context = get_experiment_context()
        if not context.get("enable_nwd_assigner", False):
            return overlaps
        nwd_ratio = float(context.get("nwd_assigner_ratio", 0.5))
        nwd = bbox_nwd_similarity(gt_bboxes, pd_bboxes, float(context.get("nwd_constant", 12.8)))
        return ((1.0 - nwd_ratio) * overlaps + nwd_ratio * nwd).clamp_(0, 1)


class ContextAwareBboxLoss:
    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        *extra_args,
        **extra_kwargs,
    ):
        loss_iou, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            *extra_args,
            **extra_kwargs,
        )
        context = get_experiment_context()
        if not fg_mask.sum():
            return loss_iou, loss_dfl

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if context.get("enable_focaler_iou", False):
            focaler_iou = focaler_iou_similarity(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                float(context.get("focaler_lower", 0.0)),
                float(context.get("focaler_upper", 0.95)),
            ).unsqueeze(-1)
            loss_iou = ((1.0 - focaler_iou) * weight).sum() / target_scores_sum
            return loss_iou, loss_dfl

        if not context.get("enable_nwd_loss", False):
            return loss_iou, loss_dfl

        nwd = bbox_nwd_similarity(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], float(context.get("nwd_constant", 12.8))
        ).unsqueeze(-1)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
        nwd_ratio = float(context.get("nwd_loss_ratio", 0.5))
        return (1.0 - nwd_ratio) * loss_iou + nwd_ratio * loss_nwd, loss_dfl


class PatchedTaskAlignedAssigner(ContextAwareTaskAlignedAssigner, tal_mod.TaskAlignedAssigner):
    """Module-level patched assigner so checkpoints remain pickleable."""


class PatchedBboxLoss(ContextAwareBboxLoss, loss_mod.BboxLoss):
    """Module-level patched bbox loss so checkpoints remain pickleable."""


def _build_parse_model():
    def parse_model(d, ch, verbose=True):
        import ast
        import contextlib as _contextlib
        import torch.nn as nn
        from ultralytics.nn.modules import (
            AIFI, C1, C2, C2PSA, C2f, C2fAttn, C2fCIB, C2fPSA, C3, C3Ghost, C3TR, C3k2, C3x,
            AConv, ADown, Bottleneck, BottleneckCSP, CBFuse, CBLinear, Classify, Concat, Conv,
            ConvTranspose, Detect, DWConv, DWConvTranspose2d, ELAN1, Focus, GhostBottleneck,
            GhostConv, HGBlock, HGStem, ImagePoolingAttn, OBB, PSA, Pose, RepC3, RepNCSPELAN4,
            RTDETRDecoder, ResNetLayer, SCDown, SPP, SPPF, SPPELAN, Segment, WorldDetect, v10Detect,
        )
        from ultralytics.nn.modules.conv import Conv as ConvClass
        from ultralytics.nn.tasks import LOGGER, colorstr, make_divisible

        legacy = True
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]
        else:
            scale = d.get("scale", "")

        if act:
            ConvClass.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        base_modules = {
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, C2fPSA,
            C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown,
            AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d,
            C3x, RepC3, PSA, SCDown, C2fCIB, DWTDown, ResidualWTConv, SPDConv, WTConv, WTRefine,
        }
        repeat_modules = {
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA,
        }

        global_namespace = globals().copy()
        global_namespace.update(locals())

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = getattr(torch.nn, m[3:]) if "nn." in m else global_namespace[m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with _contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if m is C3k2:
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
            elif m is ECABlock:
                c2 = ch[f]
                args = [c2, *args]
            elif m is DefectMorphGate:
                c2 = ch[f]
                args = [c2, *args]
            elif m is AIFI:
                args = [ch[f], *args]
            elif m in {HGStem, HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is WeightedConcat:
                c2 = sum(ch[x] for x in f)
                args = [*args, len(f)]
            elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
                detect_ch = [ch[x] for x in f]
                if m is Detect:
                    import inspect

                    detect_params = inspect.signature(Detect.__init__).parameters
                    if "reg_max" in detect_params:
                        args = [args[0], 16, False, detect_ch]
                    else:
                        args = [args[0], detect_ch]
                else:
                    args.append(detect_ch)
                if m is Segment:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in {Detect, Segment, Pose, OBB}:
                    m.legacy = legacy
            elif m is RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m is CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    return parse_model


def apply_local_ultralytics_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.DWTDown = DWTDown
    tasks.DefectMorphGate = DefectMorphGate
    tasks.ECABlock = ECABlock
    tasks.ResidualWTConv = ResidualWTConv
    tasks.SPDConv = SPDConv
    tasks.WTConv = WTConv
    tasks.WTRefine = WTRefine
    tasks.WeightedConcat = WeightedConcat
    tasks.parse_model = _build_parse_model()
    tal_mod.TaskAlignedAssigner = PatchedTaskAlignedAssigner
    loss_mod.TaskAlignedAssigner = PatchedTaskAlignedAssigner
    loss_mod.BboxLoss = PatchedBboxLoss
    tasks.v8DetectionLoss = loss_mod.v8DetectionLoss
    _PATCHED = True


def component_status(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    models_dir = repo_root / "custom_yolo" / "models"
    return {
        "dwt_module": (repo_root / "custom_yolo" / "modules.py").exists(),
        "nwd_patch": (repo_root / "custom_yolo" / "patches.py").exists(),
        "repneck_yaml": (models_dir / "yolo11n_dwt_repneck.yaml").exists(),
        "dwt_yaml": (models_dir / "yolo11n_dwt.yaml").exists(),
        "neck_wt_yaml": (models_dir / "yolo11n_neck_wt.yaml").exists(),
        "residual_wt_yaml": (models_dir / "yolo11n_residual_wt.yaml").exists(),
        "p2_yaml": (models_dir / "yolo11n_p2.yaml").exists(),
        "p2_eca_yaml": (models_dir / "yolo11n_p2_eca.yaml").exists(),
        "spd_yaml": (models_dir / "yolo11n_spd.yaml").exists(),
        "spd_defect_p2_yaml": (models_dir / "yolo11n_spd_defect_p2.yaml").exists(),
        "spd_defect_p2_dyfuse_yaml": (models_dir / "yolo11n_spd_defect_p2_dyfuse.yaml").exists(),
    }


