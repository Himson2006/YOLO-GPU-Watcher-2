import os
from dotenv import load_dotenv

load_dotenv(override=True)  # read .env

class Config:
    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    WATCH_FOLDER   = os.environ["WATCH_FOLDER"]
    YOLO_MODEL_PATH= os.environ["YOLO_MODEL_PATH"]
