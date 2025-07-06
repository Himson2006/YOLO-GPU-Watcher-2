import os, time, cv2, torch
from ultralytics import YOLO

def run_detection(
    input_source: str,
    model_path: str,
    conf_thres: float = 0.5,
    iou_thres:  float = 0.5,
    frame_threshold: int = 10,
    gap_tolerance:   int = 3,
):
    # ── wait until file stable ───────────────────────
    last, stable = -1, 0
    while stable < 2:
        try:
            sz = os.path.getsize(input_source)
        except OSError:
            time.sleep(1); continue
        if sz == last:
            stable += 1
        else:
            last, stable = sz, 0
        time.sleep(1)

    # ── load model to GPU if available ───────────────
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    if device=="cuda": torch.backends.cudnn.benchmark = True

    # ── open video & detect frame by frame ────────────
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_source!r}")

    records, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx += 1

        res = model(frame,
                    conf=conf_thres,
                    iou=iou_thres,
                    device=device,
                    half=(device=="cuda")
                   )[0]
        dets = []
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            c = float(conf)
            if c<conf_thres: continue
            x1,y1,x2,y2 = map(float, box)
            dets.append({
                "bbox": [x1,y1,x2,y2],
                "confidence": c,
                "class_id": int(cls),
                "class_name": model.names[int(cls)]
            })
        records.append({
            "frame": idx,
            "objects_detected": bool(dets),
            "num_detections": len(dets),
            "detections": dets
        })

    cap.release()
    # ── run-length filter ────────────────────────────
    class_to_frames = {}
    for rec in records:
        for d in rec["detections"]:
            class_to_frames.setdefault(d["class_name"], set()).add(rec["frame"])
    valid = {}
    for cls, frames in class_to_frames.items():
        f_sorted = sorted(frames)
        run, good = [f_sorted[0]], set()
        for f in f_sorted[1:]:
            if f - run[-1] <= gap_tolerance+1:
                run.append(f)
            else:
                if len(run) > frame_threshold: good.update(run)
                run=[f]
        if len(run)>frame_threshold: good.update(run)
        valid[cls] = good

    filtered = []
    for rec in records:
        fidx = rec["frame"]
        keep = [d for d in rec["detections"] if fidx in valid.get(d["class_name"],())]
        filtered.append({
            "frame": fidx,
            "objects_detected": bool(keep),
            "num_detections": len(keep),
            "detections": keep
        })

    return {
        "video_filename": os.path.basename(input_source),
        "total_frames": idx,
        "frames": filtered
    }
