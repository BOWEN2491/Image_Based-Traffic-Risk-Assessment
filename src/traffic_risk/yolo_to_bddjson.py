#!/usr/bin/env python3
# yolo_to_bddjson.py
# 用 YOLO 推理一张图片，输出“精简 BDD100K 风格 JSON”
# 依赖: pip install ultralytics opencv-python numpy

import os
import json
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from .config import LIGHT_CNN_DIR, YOLO_DIR

# 全局上下文：只在第一次调用时加载模型，后续复用（避免重复加载导致卡顿）
_TL_CNN_CTX = None

def _init_tl_cnn(
    model_dir= None,
    model_type="resnet18",
    img_size=128,
    device_str= None
):
    """
    一次性加载模型与类别映射；返回上下文字典。
    """
    from pathlib import Path
    model_dir = Path(model_dir)
    model_path = model_dir / "best_model.pth"
    mapping_path = model_dir / "class_indices.json"
    if not model_path.exists():
        raise FileNotFoundError(f"[TL-CNN] Model not found: {model_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"[TL-CNN] Mapping not found: {mapping_path}")

    # 设备选择
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 类别映射
    with open(mapping_path, encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # 构建与训练一致的结构（默认 resnet18；如果你训练时换了模型，这里也要相应改）
    if model_type.lower() == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("model_type 目前只支持 'resnet18'（若你训练时换了，请告诉我以便修改）")

    # 加载权重
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # 预处理
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    return {"device": device, "model": model, "tfm": tfm, "idx_to_class": idx_to_class}


# —— 你现有 build_features.py 识别的类别口径 ——
PERSON = {"person"}
RIDERS = {"rider","cyclist","bicycle","motorcycle"}      # build_features 把 rider/motorcycle/bicycle 归为 rider
VEHICLES = {"car","bus","truck","train"}
TL = {"traffic light","traffic_light","tl"}
TS = {"traffic sign","traffic_sign","sign"}

def norm(name: str) -> str:
    return name.lower().replace("_"," ").strip()

def map_to_bdd_category(name: str) -> str:
    n = norm(name)
    if n in PERSON:
        return "person"
    if n in {"bicycle"}:
        return "bicycle"
    if n in {"motorcycle"}:
        return "motorcycle"
    if n in {"rider", "cyclist"}:
        return "rider"
    if n in {"car", "bus", "truck", "train"}:
        return n
    if n in {"traffic light", "traffic_light", "tl"}:
        return "traffic light"
    if n in {"traffic sign", "traffic_sign", "sign"}:
        return "traffic sign"
    # 其它类别忽略（不写入 JSON）
    return ""

def classify_tl_color(img_bgr, box_xyxy, conf_threshold=0.50, min_side=6):
    """
    用训练好的 CNN 对交通灯 ROI 分类。
    参数:
      img_bgr: OpenCV BGR 图像 (H,W,3)
      box_xyxy: 边框 (x1,y1,x2,y2)（浮点或整型都可）
      conf_threshold: 置信度阈值，低于则返回 'unknown'
      min_side: ROI 任何一边 < min_side 则返回 'unknown'
    返回:
      'red' | 'yellow' | 'green' | 'unknown'
    """
    global _TL_CNN_CTX
    if _TL_CNN_CTX is None:
        # 第一次调用时懒加载（路径按你的模型输出目录设置）
        _TL_CNN_CTX = _init_tl_cnn(
            model_dir= LIGHT_CNN_DIR,
            model_type="resnet18",
            img_size=128,
        )

    # 越界裁剪与尺寸检查
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return "unknown"

    roi_bgr = img_bgr[y1:y2, x1:x2]
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    # BGR -> RGB -> PIL
    roi_rgb = roi_bgr[:, :, ::-1]
    pil = Image.fromarray(roi_rgb)

    # 预处理与推理
    x = _TL_CNN_CTX["tfm"](pil).unsqueeze(0).to(_TL_CNN_CTX["device"])
    with torch.no_grad():
        logits = _TL_CNN_CTX["model"](x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        conf = float(probs[pred_idx].item())
        pred_cls = _TL_CNN_CTX["idx_to_class"][pred_idx]

    if conf < conf_threshold:
        return "unknown"
    return pred_cls if pred_cls in {"red", "yellow", "green", "unknown"} else "unknown"

def _infer_and_build_json(image_path: str,
                          weights: str,
                          conf: float,
                          iou: float) -> dict:
    """
    内部工具函数：
    - 用 YOLO 推理一张图片
    - 用 TL-CNN 给交通灯补 trafficLightColor
    - 返回一个「精简 BDD100K 风格」的 dict（不写文件）
    """
    print("[DBG] loading YOLO...")
    model = YOLO(weights)
    print("[DBG] YOLO loaded")
    print("[DBG] running YOLO predict...")
    res = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)[0]
    print("[DBG] YOLO done")
    names = res.names

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    objects = []
    next_id = 0
    for b in res.boxes:
        cls_id = int(b.cls.item())
        raw_name = names.get(cls_id, str(cls_id))
        cat = map_to_bdd_category(raw_name)
        if not cat:
            continue

        x1, y1, x2, y2 = b.xyxy.cpu().numpy().reshape(-1).tolist()
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

        obj = {
            "category": cat,
            "id": next_id,
            "attributes": {
                "occluded": False,
                "truncated": False,
                # 与示例保持同名字段（trafficLightColor）
                "trafficLightColor": "unknown"  # 对非交通灯也无所谓，build_features 不会用到
            },
            "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
        }

        # 仅对交通灯补颜色（用你的 TL CNN）
        if cat == "traffic light":
            print("[DBG] init TL-CNN (lazy)...")
            color = classify_tl_color(img, (x1i, y1i, x2i, y2i), conf_threshold=0.50)
            print("[DBG] TL-CNN ok:", color)
            obj["attributes"]["trafficLightColor"] = color

        objects.append(obj)
        next_id += 1

    # 生成“精简版” BDD JSON（frames[0].objects）
    out = {
        "name": os.path.splitext(os.path.basename(image_path))[0],
        "frames": [{"timestamp": 0, "objects": objects}],
        "attributes": {
            "weather": "unknown",
            "scene": "unknown",
            "timeofday": "unknown"
        }
    }
    return out

def run_yolo_for_image(image_path: str,
                       weights: str = YOLO_DIR /"yolo11n.pt",
                       conf: float = 0.25,
                       iou: float = 0.45) -> dict:
    """
    给 pipeline / FastAPI / React 用的轻量接口：
    - 不写任何文件
    - 直接返回 BDD 风格的 dict
    """
    return _infer_and_build_json(image_path, weights, conf, iou)
