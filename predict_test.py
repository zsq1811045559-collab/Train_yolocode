from ultralytics import YOLO

# 1. 加载模型
model = YOLO("yolov8n.pt")  # 或 yolov11n.pt 等

# 2. 推理
results = model.predict(
    source="ultralytics/assets/bus.jpg",  # 图片或文件夹、视频流
    device=0,
    save=True,
)

# results=model(source="screen")
# results=model(source=0)


# 3. 显示/保存结果
# for r in results:
#     r.show()
results[0].show()  # 弹出可视化窗口
# results[0].save("steel_pre.jpg")  # 保存预测图像
