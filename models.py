from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Video(db.Model):
    __tablename__ = "video"
    id       = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String, unique=True, nullable=False)
    detections = db.relationship("Detection", backref="video", cascade="all, delete-orphan")

class Detection(db.Model):
    __tablename__ = "detection"
    id                  = db.Column(db.Integer, primary_key=True)
    video_id            = db.Column(db.Integer, db.ForeignKey("video.id"), nullable=False)
    detection_json      = db.Column(db.JSON, nullable=False)
    classes_detected    = db.Column(db.String)
    max_count_per_frame = db.Column(db.JSON)
