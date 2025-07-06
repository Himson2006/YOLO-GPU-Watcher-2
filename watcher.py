#!/usr/bin/env python
import os, time, logging
from flask import Flask
from watchdog.observers import Observer
from watchdog.events   import FileSystemEventHandler
from sqlalchemy.exc    import IntegrityError

from config    import Config
from models    import db, Video, Detection
from detection import run_detection

# ─── app & logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

watch_folder = app.config["WATCH_FOLDER"]
detect_folder= os.path.join(watch_folder, "detections")
os.makedirs(watch_folder,   exist_ok=True)
os.makedirs(detect_folder,  exist_ok=True)

class Handler(FileSystemEventHandler):
    ALLOWED = {".mp4",".avi",".mov",".mkv"}

    def on_created(self, event):
        if event.is_directory: return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext not in self.ALLOWED: return

        # scp does atomic mv → also catch moved files:
        self._process(event.src_path)

    def on_moved(self, event):
        # catch scp/rsync renames too
        if event.is_directory: return
        self.on_created(event)

    def _process(self, full_path):
        logging.info(f"[watcher] detected file: {full_path}")
        # wait until stable & then do in app context
        last, stable = -1, 0
        while stable<2:
            try: sz=os.path.getsize(full_path)
            except: time.sleep(1); continue
            if sz==last: stable+=1
            else: last,stable=sz,0
            time.sleep(1)

        filename = os.path.basename(full_path)
        with app.app_context():
            # 1) refuse dupes
            if Video.query.filter_by(filename=filename).first():
                logging.info(f"[watcher] dup '{filename}', skipping")
                return
            # 2) insert row
            vid = Video(filename=filename)
            db.session.add(vid); db.session.commit()
            logging.info(f"[watcher] row created: id={vid.id}")

            # 3) detect
            try:
                det = run_detection(full_path, app.config["YOLO_MODEL_PATH"])
            except Exception as e:
                db.session.delete(vid); db.session.commit()
                logging.error(f"[watcher] detection failed: {e}")
                return

            # 4) write JSON
            jpath = os.path.join(detect_folder, f"{os.path.splitext(filename)[0]}.json")
            with open(jpath,"w") as jf:
                import json; json.dump(det, jf, indent=2)

            # 5) summarize & save Detection row
            seen, mx = set(), {}
            for fr in det["frames"]:
                counts = {}
                for d in fr["detections"]:
                    seen.add(d["class_name"])
                    counts[d["class_name"]] = counts.get(d["class_name"],0)+1
                for c,n in counts.items():
                    mx[c] = max(mx.get(c,0), n)

            rec = Detection(
                video_id=vid.id,
                detection_json=det,
                classes_detected=",".join(sorted(seen)) or None,
                max_count_per_frame=mx or None
            )
            db.session.add(rec); db.session.commit()
            logging.info(f"[watcher] 🗸 Detection saved for vid_id={vid.id}")

    def on_deleted(self, event):
        if event.is_directory: return
        filename = os.path.basename(event.src_path)
        with app.app_context():
            vid = Video.query.filter_by(filename=filename).first()
            if not vid: return
            # delete JSON
            j = os.path.join(detect_folder, f"{os.path.splitext(filename)[0]}.json")
            if os.path.exists(j): os.remove(j)
            db.session.delete(vid); db.session.commit()
            logging.info(f"[watcher] 🗑️ Removed records for '{filename}'")

if __name__=="__main__":
    observer = Observer()
    handler  = Handler()
    observer.schedule(handler, watch_folder, recursive=False)
    observer.start()
    logging.info(f"[watcher] Watching: {watch_folder}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()