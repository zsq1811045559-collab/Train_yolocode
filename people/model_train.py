# model_train.py
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载模型（检测或分割）
    model = YOLO(r"yolo11.yaml")  # 或 yolov11n-seg.pt
    #model = YOLO(r"yolo11l.yaml")  # 或 yolov11n-seg.pt
    # 2. 开始训练
    model.train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,      # GPU:0，若无GPU则改为 "cpu"
        workers=0,     # Windows 建议设 0 避免多进程报错
        name="train_person_y11"# 输出保存到 runs/detect/train_person_y11
    )
