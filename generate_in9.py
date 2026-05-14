#!/usr/bin/env python3
"""
SAM 3 기반 IN-9 Training Dataset 생성 스크립트

원본 코드: rabbit-abacus/roleofimagebackgrounds/main.py (GrabCut, ICLR 2021)
변경점: GrabCut → SAM 3 (또는 SAM 2) bbox prompt로 FG segmentation 교체

원본 코드와의 대응:
  - 필터링 threshold 동일 (FRACTION=0.9, MIN_MASK_AMOUNT=0.1, FRACTION_2=0.5)
  - transforms 동일 (Resize(256, NEAREST) + CenterCrop(256/224))
  - compositing 로직 동일 (FG resize → bg_mask=(pixel==[0,0,0]) → combine)
  - BG 선택: 필터링 통과한 이미지에서만 선택 (원본과 동일)
  - Phase 구조: Phase 1 (필터+세그먼트+비합성 variant) → Phase 2 (mixed 합성)

원본과 다른 점 (총 18개, 상세: 논문_코드_대조.md):
  - GrabCut → SAM 3 (segmentation method)
  - val 하드코딩 → --split train/val 지원
  - 단일 synset → 9 superclass 전체 자동 처리
  - mixed_rand, mixed_next, in9l 추가 (원본 미구현)
  - fg_mask를 .npy로 저장하여 Phase 2에서 재사용 (원본은 Phase 2에서 재세그먼트)
  - BG 선택 범위: 같은 synset → 같은 superclass (논문 정의 "same class" = superclass)
  - FRACTION check: 항상 test_transform(224 crop) 사용 (원본과 동일, split 무관)

사용법:
    python generate_in9.py \\
        --in_dir /path/to/imagenet \\
        --out_dir /data/jiyoonkim/in9_sam3 \\
        --ann_dir /path/to/imagenet/annotations \\
        --split train \\
        --segmentor sam3 \\
        --sam_checkpoint /path/to/sam3_model.pt

생성되는 variant:
    original, only_bg_b, only_bg_t, only_fg, no_fg,
    mixed_same, mixed_rand, mixed_next, in9l
"""

import os
import json
import time
import random
import logging
from argparse import ArgumentParser
from collections import defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants (원본 논문/코드와 동일 — arXiv 2006.09994, Appendix A)
# ---------------------------------------------------------------------------
FRACTION = 0.9         # bbox가 crop 후 이미지 90% 이상이면 제외
MIN_MASK_AMOUNT = 0.1  # FG mask가 bbox 면적 10% 미만이면 제외
FRACTION_2 = 0.5       # crop 후 FG가 resize-only 대비 50% 미만이면 제외
GRABCUT_ITER = 5       # GrabCut iteration (원본 코드: cv2.grabCut(..., 5, ...))

TRAIN_SIZE = 256       # Resize 목표
TRAIN_CROP = 256       # Train CenterCrop
TEST_CROP = 224        # Test CenterCrop

IN9_SUPERCLASSES = {
    0: "dog",
    1: "bird",
    2: "vehicle",
    3: "reptile",
    4: "carnivore",
    5: "insect",
    6: "instrument",
    7: "primate",
    8: "fish",
}

# ---------------------------------------------------------------------------
# Transforms (원본 코드 line 10-16 과 동일)
# ---------------------------------------------------------------------------
standard_transform = transforms.Compose(
    [transforms.Resize(TRAIN_SIZE, Image.NEAREST), transforms.CenterCrop(TRAIN_CROP)]
)

test_transform = transforms.Compose(
    [transforms.Resize(TRAIN_SIZE, Image.NEAREST), transforms.CenterCrop(TEST_CROP)]
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Segmentor abstraction
# ===========================================================================

class GrabCutSegmentor:
    """
    원본 논문의 GrabCut 기반 segmentation.
    rabbit-abacus/main.py get_fg_mask() (line 50-71) 과 동일.
    """

    def __init__(self):
        pass

    def segment(self, img_np, xmin, ymin, xmax, ymax):
        """
        Args:
            img_np: (H, W, 3) uint8 RGB numpy array
            xmin, ymin, xmax, ymax: bbox (absolute pixel)
        Returns:
            fg_mask: (H, W) uint8, 0=BG 1=FG
            success: bool
        """
        mask = np.zeros(img_np.shape[:2], np.uint8)
        bgmodel = np.zeros((1, 65), np.float64)
        fgmodel = np.zeros((1, 65), np.float64)
        # 원본 코드: rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        try:
            cv2.grabCut(img_np, mask, rect, bgmodel, fgmodel,
                        GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return None, False

        # 원본 코드: np.where((mask == 2) | (mask == 0), 0, 1)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        bbox_area = (xmax - xmin) * (ymax - ymin)
        if bbox_area == 0:
            return None, False
        mask_proportion = np.sum(fg_mask) / bbox_area
        success = mask_proportion >= MIN_MASK_AMOUNT
        return fg_mask, success


class SAM3Segmentor:
    """
    SAM 3 (facebookresearch/sam3) bbox prompt 기반 segmentation.
    SAM 3 불가 시 SAM 2 (SAM2ImagePredictor)로 자동 fallback.
    """

    def __init__(self, checkpoint=None, model_cfg=None, device="cuda"):
        import torch
        self.device = device

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model(
                checkpoint=checkpoint,
                model_cfg=model_cfg,
            )
            model = model.to(device)
            self.processor = Sam3Processor(model)
            self.api_version = "sam3"
            logger.info("SAM 3 model loaded (Sam3Processor API)")
        except ImportError:
            logger.warning("sam3 package not found, trying SAM 2 fallback...")
            self._init_sam2(checkpoint, model_cfg, device)

    def _init_sam2(self, checkpoint, model_cfg, device):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(model_cfg or "sam2_hiera_l", checkpoint)
        model = model.to(device)
        self.predictor = SAM2ImagePredictor(model)
        self.api_version = "sam2"
        logger.info("SAM 2 model loaded (SAM2ImagePredictor API)")

    def segment(self, img_np, xmin, ymin, xmax, ymax):
        """
        Returns:
            fg_mask: (H, W) uint8, 0=BG 1=FG
            success: bool
        """
        import torch

        if self.api_version == "sam3":
            return self._segment_sam3(img_np, xmin, ymin, xmax, ymax)
        else:
            return self._segment_sam2(img_np, xmin, ymin, xmax, ymax)

    def _segment_sam3(self, img_np, xmin, ymin, xmax, ymax):
        import torch

        pil_img = Image.fromarray(img_np)
        inference_state = self.processor.set_image(pil_img)

        h, w = img_np.shape[:2]
        # add_geometric_prompt: normalized [cx, cy, w, h]
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        try:
            output = self.processor.add_geometric_prompt(
                box=[cx, cy, bw, bh],
                label=1,
                state=inference_state,
            )
            masks = output["masks"]   # (N, H, W) or (N, 1, H, W)
            scores = output["scores"]  # (N,)

            if masks is None or len(masks) == 0:
                return None, False

            best_idx = scores.argmax().item()
            fg_mask = masks[best_idx].cpu().numpy().astype("uint8")
            if fg_mask.ndim == 3:
                fg_mask = fg_mask.squeeze(0)

        except Exception as e:
            logger.debug(f"SAM3 segmentation failed: {e}")
            return None, False

        bbox_area = (xmax - xmin) * (ymax - ymin)
        if bbox_area == 0:
            return None, False
        mask_proportion = np.sum(fg_mask) / bbox_area
        success = mask_proportion >= MIN_MASK_AMOUNT
        return fg_mask, success

    def _segment_sam2(self, img_np, xmin, ymin, xmax, ymax):
        import torch

        self.predictor.set_image(img_np)
        input_box = np.array([xmin, ymin, xmax, ymax])

        try:
            masks, scores, _ = self.predictor.predict(
                box=input_box,
                multimask_output=True,
            )
        except Exception as e:
            logger.debug(f"SAM2 segmentation failed: {e}")
            return None, False

        if masks is None or len(masks) == 0:
            return None, False

        best_idx = np.argmax(scores)
        fg_mask = masks[best_idx].astype("uint8")

        bbox_area = (xmax - xmin) * (ymax - ymin)
        if bbox_area == 0:
            return None, False
        mask_proportion = np.sum(fg_mask) / bbox_area
        success = mask_proportion >= MIN_MASK_AMOUNT
        return fg_mask, success


# ===========================================================================
# Helper functions (원본 코드 line 20-117 기반)
# ===========================================================================

def make_if_not_exists(dirname):
    os.makedirs(dirname, exist_ok=True)


def blackout(img, mask):
    """
    원본 코드 line 28-34.
    PIL image에서 mask==255 영역을 검정으로.
    """
    np_mask = np.array(mask)
    np_img = np.array(img)
    if len(np_img.shape) == 2:
        np_img = np.stack((np_img,) * 3, axis=-1)
    np_img[np_mask == 255] = 0
    return Image.fromarray(np_img)


def combine(img1, img2, mask):
    """
    원본 코드 line 39-44.
    mask==255인 영역은 img2, 나머지는 img1.
    mask는 PIL Image 또는 numpy array 모두 가능.
    """
    np_mask = np.array(mask)
    np_img1 = np.array(img1)
    np_img2 = np.array(img2)
    np_img1[np_mask == 255] = np_img2[np_mask == 255]
    return Image.fromarray(np_img1)


def get_bg_tiled(img, xmin, ymin, xmax, ymax):
    """
    원본 코드 line 76-107.
    bbox 외 최대 배경 영역을 tile로 반복.
    """
    width, height = img.size
    xmin_margin = xmin
    ymin_margin = ymin
    xmax_margin = width - xmax
    ymax_margin = height - ymax
    max_x_area = max(xmin_margin, xmax_margin) * height
    max_y_area = max(ymin_margin, ymax_margin) * width

    if max(max_x_area, max_y_area) <= 0:
        raise ValueError("No background rectangles left in this image")

    use_horizontal = max_x_area > max_y_area
    if use_horizontal:
        tile_ymin, tile_ymax = 0, height
        if xmin_margin > xmax_margin:
            tile_xmin, tile_xmax = 0, xmin
        else:
            tile_xmin, tile_xmax = xmax, width
    else:
        tile_xmin, tile_xmax = 0, width
        if ymin_margin > ymax_margin:
            tile_ymin, tile_ymax = 0, ymin
        else:
            tile_ymin, tile_ymax = ymax, height

    tile = img.crop([tile_xmin, tile_ymin, tile_xmax, tile_ymax])
    tile_w, tile_h = tile.size
    bg_tiled = Image.new("RGB", (width, height))
    for i in range(0, width, tile_w):
        for j in range(0, height, tile_h):
            bg_tiled.paste(tile, (i, j))
    return bg_tiled


def parse_annotation(ann_path):
    """PASCAL VOC XML annotation 파싱."""
    tree = ET.parse(ann_path)
    root = tree.getroot()

    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        bboxes.append((xmin, ymin, xmax, ymax))

    return width, height, bboxes


def load_in_to_in9_mapping(json_path):
    """in_to_in9.json: {ImageNet class index (str) → IN-9 superclass index (int)}."""
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_idx_to_synset(in_dir, split="train"):
    """
    ImageNet 디렉토리의 synset 폴더 → class index 매핑.
    sorted 순서 = torchvision ImageFolder 순서 = ImageNet class index.
    """
    split_dir = os.path.join(in_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"ImageNet {split} directory not found: {split_dir}")

    synsets = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])
    return {i: s for i, s in enumerate(synsets)}


def find_annotation_for_image(ann_dir, split, synset, img_filename):
    """
    ImageNet annotation XML 경로 탐색.
    Train: {ann_dir}/{synset}/{stem}.xml 또는 {ann_dir}/train/{synset}/{stem}.xml
    Val:   {ann_dir}/val/{stem}.xml
    """
    img_stem = os.path.splitext(img_filename)[0]

    if split == "train":
        candidates = [
            os.path.join(ann_dir, synset, f"{img_stem}.xml"),
            os.path.join(ann_dir, "train", synset, f"{img_stem}.xml"),
        ]
    else:
        candidates = [
            os.path.join(ann_dir, "val", f"{img_stem}.xml"),
            os.path.join(ann_dir, f"{img_stem}.xml"),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ===========================================================================
# Filtering (원본 코드 line 121-186 과 동일한 로직)
# ===========================================================================

def is_good_image(img_pil, ann_width, ann_height, bboxes, split):
    """
    원본 코드의 is_good_image() 와 동일한 필터링 로직.

    Returns:
        0: 사용 가능
        3: annotation과 이미지 dimension 불일치
        4: bbox가 정확히 1개가 아님
        5: bbox가 crop 후 이미지의 FRACTION(90%) 이상 차지
        6: crop 후 bbox 면적이 resize-only 대비 FRACTION_2(50%) 미만
    """
    width, height = img_pil.size

    if ann_width != width or ann_height != height:
        return 3

    if len(bboxes) != 1:
        return 4

    xmin, ymin, xmax, ymax = bboxes[0]

    # bbox 영역 mask 생성
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([xmin, ymin, xmax, ymax], fill="white")

    # FRACTION check (원본 코드 line 167-171)
    # 원본은 항상 test_transform(224 crop)을 사용.
    # 이유: 최종 평가 시 224 crop 기준으로 bbox 비율을 체크하므로 split 무관.
    post_crop = test_transform(mask)
    num_masked_pixels = np.sum(np.array(post_crop)) // (255 * 3)
    area_ratio = num_masked_pixels / (TEST_CROP ** 2)
    if area_ratio > FRACTION:
        return 5

    # FRACTION_2 check (원본 코드 line 176-183)
    post_resize_only = transforms.Resize(TRAIN_SIZE, Image.NEAREST)(mask)
    num_masked_resize = np.sum(np.array(post_resize_only)) // (255 * 3)
    if num_masked_pixels < FRACTION_2 * num_masked_resize:
        return 6

    return 0


# ===========================================================================
# Phase 1: 필터링 + 세그먼트 + 비합성 variant 저장
# ===========================================================================

def phase1_process_superclass(
    superclass_idx, synset_list, in_dir, out_dir, ann_dir, split, segmentor
):
    """
    원본 코드 Phase 1 (line 226-293) + Phase 2의 only_fg/no_fg (line 356-358) 에 대응.

    원본은 Phase 2에서 GrabCut을 재실행하여 only_fg/no_fg를 생성했는데,
    이는 JPEG 압축 아티팩트 회피 목적이다 (원본 코드 line 297-298 주석 참조).
    우리는 fg_mask를 .npy로 저장하므로 JPEG 문제가 없어 Phase 1에서 한 번에 처리.

    Returns:
        good_images: list of dict (Phase 2에서 사용할 메타데이터)
        stats: dict
    """
    sc_name = IN9_SUPERCLASSES[superclass_idx]
    sc_label = str(superclass_idx)
    split_dir = os.path.join(in_dir, split)

    stats = defaultdict(int)
    good_images = []

    logger.info(f"[SC {superclass_idx} ({sc_name})] Phase 1: filtering + segmentation")

    for synset in synset_list:
        synset_dir = os.path.join(split_dir, synset)
        if not os.path.isdir(synset_dir):
            logger.warning(f"  synset dir not found: {synset_dir}")
            continue

        img_files = sorted([
            f for f in os.listdir(synset_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ])
        stats["total_images"] += len(img_files)

        for img_file in img_files:
            img_path = os.path.join(synset_dir, img_file)
            ann_path = find_annotation_for_image(ann_dir, split, synset, img_file)

            if ann_path is None:
                stats["no_annotation"] += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                stats["load_error"] += 1
                continue

            try:
                ann_w, ann_h, bboxes = parse_annotation(ann_path)
            except Exception:
                stats["ann_parse_error"] += 1
                continue

            # 필터링 (원본 코드 is_good_image와 동일)
            check = is_good_image(img, ann_w, ann_h, bboxes, split)
            if check != 0:
                stats[f"filter_err_{check}"] += 1
                continue

            xmin, ymin, xmax, ymax = bboxes[0]

            # FG segmentation
            img_np = np.array(img)
            fg_mask, success = segmentor.segment(img_np, xmin, ymin, xmax, ymax)
            if not success:
                stats["seg_fail"] += 1
                continue

            stats["accepted"] += 1

            # 파일명: 원본 파일명 그대로 사용
            # (train: n02279972_1234.JPEG, val: ILSVRC2012_val_00001234.JPEG)
            out_filename = img_file

            # --- original (원본 코드 line 253, 286-290) ---
            path = os.path.join(out_dir, "original", split, sc_label)
            make_if_not_exists(path)
            img.save(os.path.join(path, out_filename))

            # --- only_bg_b (원본 코드 line 274) ---
            bbox_mask = Image.new("RGB", img.size, (0, 0, 0))
            draw = ImageDraw.Draw(bbox_mask)
            draw.rectangle([xmin, ymin, xmax, ymax], fill="white")
            bg_b = blackout(img, bbox_mask)
            path = os.path.join(out_dir, "only_bg_b", split, sc_label)
            make_if_not_exists(path)
            bg_b.save(os.path.join(path, out_filename))

            # --- only_bg_t (원본 코드 line 275-276) ---
            try:
                bg_tiled = get_bg_tiled(img, xmin, ymin, xmax, ymax)
                bg_t = combine(img, bg_tiled, bbox_mask)
            except ValueError:
                bg_t = bg_b  # tiling 불가 시 blackout과 동일
            path = os.path.join(out_dir, "only_bg_t", split, sc_label)
            make_if_not_exists(path)
            bg_t.save(os.path.join(path, out_filename))

            # --- only_fg (원본 코드 line 357) ---
            only_fg_img = Image.fromarray(img_np * fg_mask[:, :, np.newaxis])
            path = os.path.join(out_dir, "only_fg", split, sc_label)
            make_if_not_exists(path)
            only_fg_img.save(os.path.join(path, out_filename))

            # --- no_fg (원본 코드 line 358) ---
            no_fg_img = Image.fromarray(img_np * (1 - fg_mask[:, :, np.newaxis]))
            path = os.path.join(out_dir, "no_fg", split, sc_label)
            make_if_not_exists(path)
            no_fg_img.save(os.path.join(path, out_filename))

            # --- fg_mask .npy 저장 ---
            # 원본 코드는 Phase 2에서 GrabCut을 재실행했으나 (JPEG 압축 회피),
            # 우리는 .npy로 저장하여 lossless 보존.
            stem = os.path.splitext(img_file)[0]
            mask_dir = os.path.join(out_dir, "fg_mask", split, sc_label)
            make_if_not_exists(mask_dir)
            fg_mask_path = os.path.join(mask_dir, f"{stem}.npy")
            np.save(fg_mask_path, fg_mask)

            # Phase 2 용 메타데이터 수집
            good_images.append({
                "synset": synset,
                "img_file": img_file,
                "img_path": img_path,
                "ann_path": ann_path,
                "bbox": (xmin, ymin, xmax, ymax),
                "fg_mask_path": fg_mask_path,
                "out_filename": out_filename,
            })

    logger.info(
        f"[SC {superclass_idx} ({sc_name})] Phase 1 done: "
        f"{stats['accepted']}/{stats['total_images']} accepted "
        f"(no_ann={stats['no_annotation']}, seg_fail={stats['seg_fail']}, "
        f"filter3={stats.get('filter_err_3',0)}, filter4={stats.get('filter_err_4',0)}, "
        f"filter5={stats.get('filter_err_5',0)}, filter6={stats.get('filter_err_6',0)})"
    )

    return good_images, dict(stats)


# ===========================================================================
# Phase 2: Mixed variant 합성
# ===========================================================================

def _pick_bg_image(bg_sc, all_good_images):
    """
    BG 이미지를 필터링 통과한 이미지(good_images)에서 랜덤 선택 후
    Only-BG-T 처리하여 반환.

    원본 코드 (line 365-400) 와 동일한 로직:
      1. out_dir/original/ 에 있는 (=필터링 통과한) 이미지에서 랜덤 선택
      2. 원본 이미지 로드 → bbox로 BG tiling → Only-BG-T 생성

    Returns:
        (only_bg_t_image, bg_synset, bg_stem) or (None, None, None)
    """
    candidates = all_good_images.get(bg_sc, [])
    if not candidates:
        return None, None, None

    max_attempts = 10
    for _ in range(max_attempts):
        chosen = random.choice(candidates)
        try:
            img = Image.open(chosen["img_path"]).convert("RGB")
            xmin, ymin, xmax, ymax = chosen["bbox"]

            # bbox mask → BG tiling (원본 코드 line 388-394)
            bbox_mask = Image.new("RGB", img.size, (0, 0, 0))
            draw = ImageDraw.Draw(bbox_mask)
            draw.rectangle([xmin, ymin, xmax, ymax], fill="white")

            bg_tiled = get_bg_tiled(img, xmin, ymin, xmax, ymax)
            only_bg_t = combine(img, bg_tiled, bbox_mask)

            bg_stem = os.path.splitext(chosen["out_filename"])[0]
            return only_bg_t, chosen["synset"], bg_stem

        except Exception:
            continue

    return None, None, None


def phase2_generate_mixed(
    superclass_idx, good_images, all_good_images, out_dir, split
):
    """
    원본 코드 Phase 2 (line 299-417) 에 대응.
    mixed_same + mixed_rand + mixed_next 생성.

    원본 코드와의 대응:
      - fg_mask 재사용: 원본은 GrabCut 재실행 (line 343-346),
        우리는 Phase 1에서 저장한 .npy 로드 (JPEG 문제 없음)
      - FG 합성: 원본과 동일 (line 357→resize→bg_mask→combine, line 396-397)
      - BG 선택: 필터링 통과 이미지에서 선택 (원본 line 365-366과 동일)
      - 파일명: fg_{fg_stem}_bg_{bg_stem}.JPEG (원본 line 414)
    """
    sc_name = IN9_SUPERCLASSES[superclass_idx]
    sc_label = str(superclass_idx)

    next_sc = (superclass_idx + 1) % 9
    other_scs = [i for i in range(9) if i != superclass_idx]

    stats = defaultdict(int)

    logger.info(
        f"[SC {superclass_idx} ({sc_name})] Phase 2: "
        f"mixed variants ({len(good_images)} images)"
    )

    for img_info in good_images:
        # fg_mask 로드 (원본은 여기서 GrabCut 재실행)
        fg_mask = np.load(img_info["fg_mask_path"])

        # FG 이미지 생성 + resize (원본 코드 line 356-361)
        img = Image.open(img_info["img_path"]).convert("RGB")
        img_np = np.array(img)
        only_fg = Image.fromarray(img_np * fg_mask[:, :, np.newaxis])
        fg_resized = standard_transform(only_fg)

        # bg_mask: FG resize 후 검정 영역 = BG가 들어갈 자리
        # 원본 코드 line 361:
        #   bg_mask = np.all(np.array(fg_resized) == [0, 0, 0], axis=-1) * 255
        fg_arr = np.array(fg_resized)
        bg_mask_2d = np.all(fg_arr == [0, 0, 0], axis=-1).astype(np.uint8) * 255

        fg_stem = os.path.splitext(img_info["out_filename"])[0]

        # --- mixed variant 별 BG superclass ---
        mixed_configs = {
            "mixed_same": superclass_idx,
            "mixed_rand": random.choice(other_scs),
            "mixed_next": next_sc,
        }

        for variant_name, bg_sc in mixed_configs.items():
            bg_img, bg_synset, bg_stem = _pick_bg_image(bg_sc, all_good_images)
            if bg_img is None:
                stats[f"{variant_name}_bg_fail"] += 1
                continue

            # BG resize + 합성 (원본 코드 line 395-397)
            bg_resized = standard_transform(bg_img)
            mixed = combine(fg_resized, bg_resized, bg_mask_2d)

            # 파일명 (원본 코드 line 413-414):
            #   fg_{synset}_{image_num}_bg_{bg_synset}_{bg_image_num}.JPEG
            mixed_filename = f"fg_{fg_stem}_bg_{bg_stem}.JPEG"

            path = os.path.join(out_dir, variant_name, split, sc_label)
            make_if_not_exists(path)
            mixed.save(os.path.join(path, mixed_filename))

            stats[f"{variant_name}_ok"] += 1

    logger.info(
        f"[SC {superclass_idx} ({sc_name})] Phase 2 done: "
        f"same={stats.get('mixed_same_ok',0)}, "
        f"rand={stats.get('mixed_rand_ok',0)}, "
        f"next={stats.get('mixed_next_ok',0)}"
    )

    return dict(stats)


# ===========================================================================
# IN-9L 생성
# ===========================================================================

def generate_in9l(in_dir, out_dir, in_to_in9, idx_to_synset, split,
                  target_per_class=20000):
    """
    IN-9L: GrabCut/SAM 없이 balanced sampling.
    원본 ~180K = 20,000/class × 9 classes.
    """
    logger.info(f"Generating IN-9L ({split})...")

    sc_synsets = defaultdict(list)
    for cls_idx, sc_idx in in_to_in9.items():
        if sc_idx == -1:
            continue
        synset = idx_to_synset.get(cls_idx)
        if synset is not None:
            sc_synsets[sc_idx].append(synset)

    split_dir = os.path.join(in_dir, split)

    for sc_idx in range(9):
        synsets = sc_synsets[sc_idx]
        sc_name = IN9_SUPERCLASSES[sc_idx]
        sc_label = str(sc_idx)

        all_images = []
        for synset in synsets:
            synset_dir = os.path.join(split_dir, synset)
            if not os.path.isdir(synset_dir):
                continue
            for f in os.listdir(synset_dir):
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    all_images.append(os.path.join(synset_dir, f))

        if len(all_images) > target_per_class:
            selected = random.sample(all_images, target_per_class)
        else:
            selected = all_images
            logger.warning(
                f"  IN-9L SC {sc_idx} ({sc_name}): only {len(selected)} images "
                f"(target={target_per_class})"
            )

        out_path = os.path.join(out_dir, "in9l", split, sc_label)
        make_if_not_exists(out_path)

        for img_path in selected:
            basename = os.path.basename(img_path)
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized = standard_transform(img)
                img_resized.save(os.path.join(out_path, basename))
            except Exception:
                continue

        logger.info(f"  IN-9L SC {sc_idx} ({sc_name}): {len(selected)} images saved")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = ArgumentParser(description="SAM 3 기반 IN-9 dataset 생성")
    parser.add_argument("--in_dir", required=True,
                        help="ImageNet 루트 (하위에 train/, val/ 존재)")
    parser.add_argument("--out_dir", required=True,
                        help="출력 디렉토리")
    parser.add_argument("--ann_dir", required=True,
                        help="ImageNet bbox annotation 디렉토리")
    parser.add_argument("--mapping_json", default=None,
                        help="in_to_in9.json 경로 (기본: 스크립트와 같은 디렉토리)")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="train 또는 val")
    parser.add_argument("--segmentor", default="sam3", choices=["sam3", "grabcut"],
                        help="FG segmentor: sam3 (기본) 또는 grabcut (원본 재현)")
    parser.add_argument("--sam_checkpoint", default=None,
                        help="SAM 모델 체크포인트 경로")
    parser.add_argument("--sam_model_cfg", default=None,
                        help="SAM 모델 config (e.g. sam3_hiera_large)")
    parser.add_argument("--device", default="cuda",
                        help="SAM 디바이스 (cuda/cpu)")
    parser.add_argument("--superclasses", default=None, type=str,
                        help="처리할 superclass (쉼표 구분, e.g. '0,1,2'). 미지정 시 전체.")
    parser.add_argument("--skip_mixed", action="store_true",
                        help="mixed variant 생성 건너뛰기 (Phase 2 skip)")
    parser.add_argument("--skip_in9l", action="store_true",
                        help="IN-9L 생성 건너뛰기")
    parser.add_argument("--in9l_per_class", type=int, default=20000,
                        help="IN-9L 클래스당 이미지 수 (기본: 20000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- in_to_in9.json 로드 ---
    mapping_path = args.mapping_json or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "in_to_in9.json"
    )
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"in_to_in9.json not found at {mapping_path}. "
            f"Use --mapping_json to specify path."
        )
    in_to_in9 = load_in_to_in9_mapping(mapping_path)
    logger.info(f"Loaded in_to_in9 mapping: {len(in_to_in9)} entries")

    # --- ImageNet synset ↔ index 매핑 ---
    idx_to_synset = build_idx_to_synset(args.in_dir, args.split)
    logger.info(f"Found {len(idx_to_synset)} synsets in {args.in_dir}/{args.split}")

    # --- Superclass별 synset 리스트 ---
    all_superclass_synsets = defaultdict(list)
    for cls_idx, sc_idx in in_to_in9.items():
        if sc_idx == -1:
            continue
        synset = idx_to_synset.get(cls_idx)
        if synset is not None:
            all_superclass_synsets[sc_idx].append(synset)

    for sc_idx in range(9):
        logger.info(
            f"  SC {sc_idx} ({IN9_SUPERCLASSES[sc_idx]}): "
            f"{len(all_superclass_synsets[sc_idx])} synsets"
        )

    # --- Segmentor 초기화 ---
    if args.segmentor == "sam3":
        segmentor = SAM3Segmentor(
            checkpoint=args.sam_checkpoint,
            model_cfg=args.sam_model_cfg,
            device=args.device,
        )
    else:
        segmentor = GrabCutSegmentor()

    logger.info(f"Segmentor: {args.segmentor}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {args.out_dir}")

    # --- 처리할 superclass ---
    if args.superclasses:
        sc_indices = [int(x.strip()) for x in args.superclasses.split(",")]
    else:
        sc_indices = list(range(9))

    # ===================================================================
    # Phase 1: 전체 superclass 필터링 + 세그먼트 + 비합성 variant
    # (원본 코드 Phase 1 + Phase 2의 only_fg/no_fg에 대응)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Filter + Segment + Save non-mixed variants")
    logger.info("=" * 60)

    all_good_images = {}  # {sc_idx: [good_image_info]}
    all_stats = {}

    for sc_idx in sc_indices:
        synset_list = all_superclass_synsets[sc_idx]
        if not synset_list:
            logger.warning(f"SC {sc_idx}: no synsets found, skipping")
            continue

        t0 = time.time()
        good_images, stats = phase1_process_superclass(
            superclass_idx=sc_idx,
            synset_list=synset_list,
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            ann_dir=args.ann_dir,
            split=args.split,
            segmentor=segmentor,
        )
        elapsed = time.time() - t0
        all_good_images[sc_idx] = good_images
        all_stats[sc_idx] = stats
        logger.info(
            f"SC {sc_idx} Phase 1 done: {len(good_images)} images, {elapsed:.1f}s"
        )

    # ===================================================================
    # Phase 2: Mixed variant 합성
    # (원본 코드 Phase 2의 mixed_same에 대응 + mixed_rand/next 추가)
    #
    # Phase 1이 전체 superclass에 대해 완료된 후 실행 →
    # mixed_rand/next에서 다른 superclass의 good_images 사용 가능.
    # ===================================================================
    if not args.skip_mixed:
        logger.info("=" * 60)
        logger.info("PHASE 2: Generate mixed variants")
        logger.info("=" * 60)

        for sc_idx in sc_indices:
            good_images = all_good_images.get(sc_idx, [])
            if not good_images:
                continue

            t0 = time.time()
            mixed_stats = phase2_generate_mixed(
                superclass_idx=sc_idx,
                good_images=good_images,
                all_good_images=all_good_images,
                out_dir=args.out_dir,
                split=args.split,
            )
            elapsed = time.time() - t0
            # Phase 1 stats에 병합
            all_stats[sc_idx].update(mixed_stats)
            logger.info(f"SC {sc_idx} Phase 2 done: {elapsed:.1f}s")

    # ===================================================================
    # IN-9L
    # ===================================================================
    if not args.skip_in9l:
        logger.info("=" * 60)
        logger.info("IN-9L: Balanced sampling (no segmentation)")
        logger.info("=" * 60)
        generate_in9l(
            args.in_dir, args.out_dir, in_to_in9, idx_to_synset,
            args.split, args.in9l_per_class,
        )

    # ===================================================================
    # 최종 통계
    # ===================================================================
    logger.info("=" * 60)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 60)
    total_accepted = 0
    total_images = 0
    for sc_idx in sorted(all_stats.keys()):
        s = all_stats[sc_idx]
        accepted = s.get("accepted", 0)
        total = s.get("total_images", 0)
        total_accepted += accepted
        total_images += total
        logger.info(
            f"  SC {sc_idx} ({IN9_SUPERCLASSES[sc_idx]:>10s}): "
            f"{accepted:>6d} / {total:>6d} accepted "
            f"(no_ann={s.get('no_annotation',0)}, "
            f"seg_fail={s.get('seg_fail',0)}, "
            f"filter4={s.get('filter_err_4',0)}, "
            f"filter5={s.get('filter_err_5',0)}, "
            f"filter6={s.get('filter_err_6',0)})"
        )
    logger.info(f"  TOTAL: {total_accepted} / {total_images} accepted")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
