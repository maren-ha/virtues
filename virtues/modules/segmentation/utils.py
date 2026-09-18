from __future__ import annotations
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from instanseg.utils.tiling import _chops, _stitch, _stitch_mean, _tiles_from_chops
from itertools import product

def _chop_start(x):
    if isinstance(x, slice):
        return int(x.start or 0)
    try:
        return int(x[0])
    except (TypeError, IndexError):
        return int(x)


def _chop_starts(chops):
    return [_chop_start(chop) for chop in chops]


def _chop_top_left_coords(chop_idx):
    y_starts = _chop_starts(chop_idx[-2])
    x_starts = _chop_starts(chop_idx[-1])
    return [(y, x) for y, x in product(y_starts, x_starts)]

def segment_large_tissue(
    image: torch.Tensor,
    segmentation_model: nn.Module,
    channel_ids: torch.Tensor,
    tile: int,
    ovlp: int,
    bs: int,
    device: str = "cuda",
    *,
    max_seeds: int = 10000,
    window_size: int = 64,
    detection_size: int = 20,
    return_patch_embeddings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict a full-tissue instance mask and semantic logits for tissues too large to
    segment in one shot.

    Args:
        image: Input image of shape (C, H, W).
        segmentation_model: A segmentation model.
        channel_ids: A tensor of shape (C,) containing the channel IDs for the input image.
        tile: The size of the tiles to use for segmentation.
        ovlp: The amount of overlap between tiles.
        bs: The batch size to use for segmentation.
        device: The device to use for segmentation. Defaults to "cuda".
        max_seeds: The maximum number of seeds to use for instance segmentation. Defaults to 10000.
        window_size: The window size to use for instance segmentation. Defaults to 64.
        detection_size: The detection size to use for instance segmentation. Defaults to 20.
    Returns:
        A tuple containing the predicted instance mask and semantic logits.
    """
    h, w = int(image.shape[-2]), int(image.shape[-1])
    tile_hw = (min(tile, h), min(tile, w))
    chop_idx = _chops(image.shape, shape=tile_hw, overlap=2 * (ovlp + detection_size))
    tiles = _tiles_from_chops(image, shape=tile_hw, tuple_index=chop_idx)
    patch_coords_all = _chop_top_left_coords(chop_idx)

    instance_processor = segmentation_model.instance_processor
    n_instance_channels = int(segmentation_model.dim_out)

    instance_label_tiles = []
    semantic_logit_tiles = []
    patch_embedding_tiles = []
    patch_coords_yx = []

    with torch.no_grad():
        for i in tqdm(range(0, len(tiles), bs)):
            batch = torch.stack(tiles[i : i + bs]).to(device)
            channels = [channel_ids.to(device)] * len(batch)
            if return_patch_embeddings:
                logits, patch_embeddings = segmentation_model([img for img in batch], channels, return_patch_embeddings=True)
                patch_embedding_tiles.extend([p.detach().cpu() for p in patch_embeddings])
                patch_coords_yx.extend(patch_coords_all[i : i + len(batch)])
            else:
                logits = segmentation_model([img for img in batch], channels)
            logits = logits.detach()
            if logits.shape[-2:] != tile_hw:
                logits = F.interpolate(logits, size=tile_hw, mode="bilinear", align_corners=False)

            for tile_logits in logits:
                instance_label = instance_processor.postprocessing(
                    tile_logits[:n_instance_channels],
                    max_seeds=max_seeds,
                    window_size=window_size,
                    cleanup_fragments=True,
                )
                instance_label_tiles.append(instance_label.cpu())
                semantic_logit_tiles.append(tile_logits[n_instance_channels:].cpu())

    pred_instance, _ = _stitch(
        instance_label_tiles, shape=tile_hw, chop_list=chop_idx, offset=ovlp, final_shape=(1, h, w),
    )
    semantic_logits = _stitch_mean(
        semantic_logit_tiles,
        shape=tile_hw,
        chop_list=chop_idx,
        final_shape=(semantic_logit_tiles[0].shape[0], h, w),
    )
    if return_patch_embeddings:
        return pred_instance[0], semantic_logits, torch.stack(patch_embedding_tiles, dim=0), torch.tensor(patch_coords_yx, dtype=torch.long)
    return pred_instance[0], semantic_logits
