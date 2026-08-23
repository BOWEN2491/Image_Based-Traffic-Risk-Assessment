#!/usr/bin/env python3
# yolo_to_bddjson.py
# ç”¨ YOLO æŽ¨ç†ä¸€å¼ å›¾ç‰‡ï¼Œè¾“å‡ºâ€œç²¾ç®€ BDD100K é£Žæ ¼ JSONâ€
# ä¾èµ–: pip install ultralytics opencv-python numpy

import argparse, os, json
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from .config import LIGHT_CNN_DIR, YOLO_DIR

# å…¨å±€ä¸Šä¸‹æ–‡ï¼šåªåœ¨ç¬¬ä¸€æ¬¡è°ƒç”¨æ—¶åŠ è½½æ¨¡åž‹ï¼ŒåŽç»­å¤ç”¨ï¼ˆé¿å…é‡å¤åŠ è½½å¯¼è‡´å¡é¡¿ï¼‰
_TL_CNN_CTX = None

def _init_tl_cnn(
    model_dir= None,
    model_type="resnet18",
    img_size=128,
    device_str= None
):
    """
    ä¸€æ¬¡æ€§åŠ è½½æ¨¡åž‹ä¸Žç±»åˆ«æ˜ å°„ï¼›è¿”å›žä¸Šä¸‹æ–‡å­—å…¸ã€‚
    """
    from pathlib import Path
    import json
    model_dir = Path(model_dir)
    model_path = model_dir / "best_model.pth"
    mapping_path = model_dir / "class_indices.json"
    if not model_path.exists():
        raise FileNotFoundError(f"[TL-CNN] Model not found: {model_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"[TL-CNN] Mapping not found: {mapping_path}")

    # è®¾å¤‡é€‰æ‹©
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ç±»åˆ«æ˜ å°„
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # æž„å»ºä¸Žè®­ç»ƒä¸€è‡´çš„ç»“æž„ï¼ˆé»˜è®¤ resnet18ï¼›å¦‚æžœä½ è®­ç»ƒæ—¶æ¢äº†æ¨¡åž‹ï¼Œè¿™é‡Œä¹Ÿè¦ç›¸åº”æ”¹ï¼‰
    if model_type.lower() == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("model_type ç›®å‰åªæ”¯æŒ 'resnet18'ï¼ˆè‹¥ä½ è®­ç»ƒæ—¶æ¢äº†ï¼Œè¯·å‘Šè¯‰æˆ‘ä»¥ä¾¿ä¿®æ”¹ï¼‰")

    # åŠ è½½æƒé‡
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # é¢„å¤„ç†
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    return {"device": device, "model": model, "tfm": tfm, "idx_to_class": idx_to_class}


# â€”â€” ä½ çŽ°æœ‰ build_features.py è¯†åˆ«çš„ç±»åˆ«å£å¾„ â€”â€”
PERSON = {"person"}
RIDERS = {"rider","cyclist","bicycle","motorcycle"}      # build_features æŠŠ rider/motorcycle/bicycle å½’ä¸º rider
VEHICLES = {"car","bus","truck","train"}
TL = {"traffic light","traffic_light","tl"}
TS = {"traffic sign","traffic_sign","sign"}

def norm(name: str) -> str:
    return name.lower().replace("_"," ").strip()

def map_to_bdd_category(name: str) -> str:
    n = norm(name)
    if n in PERSON: return "person"
    if n in {"bicycle"}: return "bicycle"        # ç»™ rider è®¡æ•°å¤‡ç”¨
    if n in {"motorcycle"}: return "motorcycle"
    if n in {"rider","cyclist"}: return "rider"
    if n in {"car","bus","truck","train"}: return n
    if n in {"traffic light","traffic_light","tl"}: return "traffic light"
    if n in {"traffic sign","traffic_sign","sign"}: return "traffic sign"
    # å…¶å®ƒç±»åˆ«å¿½ç•¥ï¼ˆä¸å†™å…¥ JSONï¼‰
    return ""

def classify_tl_color(img_bgr, box_xyxy, conf_threshold=0.50, min_side=6):
    """
    ç”¨è®­ç»ƒå¥½çš„ CNN å¯¹äº¤é€šç¯ ROI åˆ†ç±»ã€‚
    å‚æ•°:
      img_bgr: OpenCV BGR å›¾åƒ (H,W,3)
      box_xyxy: è¾¹æ¡† (x1,y1,x2,y2)ï¼ˆæµ®ç‚¹æˆ–æ•´åž‹éƒ½å¯ï¼‰
      conf_threshold: ç½®ä¿¡åº¦é˜ˆå€¼ï¼Œä½ŽäºŽåˆ™è¿”å›ž 'unknown'
      min_side: ROI ä»»ä½•ä¸€è¾¹ < min_side åˆ™è¿”å›ž 'unknown'
    è¿”å›ž:
      'red' | 'yellow' | 'green' | 'unknown'
    """
    global _TL_CNN_CTX
    if _TL_CNN_CTX is None:
        # ç¬¬ä¸€æ¬¡è°ƒç”¨æ—¶æ‡’åŠ è½½ï¼ˆè·¯å¾„æŒ‰ä½ çš„æ¨¡åž‹è¾“å‡ºç›®å½•è®¾ç½®ï¼‰
        _TL_CNN_CTX = _init_tl_cnn(
            model_dir= LIGHT_CNN_DIR,
            model_type="resnet18",
            img_size=128,
        )

    # è¶Šç•Œè£å‰ªä¸Žå°ºå¯¸æ£€æŸ¥
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
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

    # é¢„å¤„ç†ä¸ŽæŽ¨ç†
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
    å†…éƒ¨å·¥å…·å‡½æ•°ï¼š
    - ç”¨ YOLO æŽ¨ç†ä¸€å¼ å›¾ç‰‡
    - ç”¨ TL-CNN ç»™äº¤é€šç¯è¡¥ trafficLightColor
    - è¿”å›žä¸€ä¸ªã€Œç²¾ç®€ BDD100K é£Žæ ¼ã€çš„ dictï¼ˆä¸å†™æ–‡ä»¶ï¼‰
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
                # ä¸Žç¤ºä¾‹ä¿æŒåŒåå­—æ®µï¼ˆtrafficLightColorï¼‰
                "trafficLightColor": "unknown"  # å¯¹éžäº¤é€šç¯ä¹Ÿæ— æ‰€è°“ï¼Œbuild_features ä¸ä¼šç”¨åˆ°
            },
            "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
        }

        # ä»…å¯¹äº¤é€šç¯è¡¥é¢œè‰²ï¼ˆç”¨ä½ çš„ TL CNNï¼‰
        if cat == "traffic light":
            print("[DBG] init TL-CNN (lazy)...")
            color = classify_tl_color(img, (x1i, y1i, x2i, y2i), conf_threshold=0.50)
            print("[DBG] TL-CNN ok:", color)
            obj["attributes"]["trafficLightColor"] = color

        objects.append(obj)
        next_id += 1

    # ç”Ÿæˆâ€œç²¾ç®€ç‰ˆâ€ BDD JSONï¼ˆframes[0].objectsï¼‰
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
    ç»™ pipeline / FastAPI / React ç”¨çš„è½»é‡æŽ¥å£ï¼š
    - ä¸å†™ä»»ä½•æ–‡ä»¶
    - ç›´æŽ¥è¿”å›ž BDD é£Žæ ¼çš„ dict
    """
    return _infer_and_build_json(image_path, weights, conf, iou)

