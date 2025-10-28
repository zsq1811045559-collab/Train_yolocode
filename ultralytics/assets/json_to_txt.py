import json, os
from pathlib import Path

json_dir = Path(r"E:\python project\Yolo_learning\ultralytics-main\ultralytics\assets")
out_dir = Path(r"E:\python project\Yolo_learning\ultralytics-main\ultralytics\assets\txts"); out_dir.mkdir(exist_ok=True)
for jpath in json_dir.glob("*.json"):
    data = json.load(open(jpath, encoding="utf-8"))
    W, H = data["imageWidth"], data["imageHeight"]
    out = ""
    for shp in data["shapes"]:
        pts = [(x/W, y/H) for x, y in shp["points"]]
        cls = 0  # 或按类别名映射
        out += f"{cls} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in pts]) + "\n"
    open(out_dir / f"{jpath.stem}.txt", "w", encoding="utf-8").write(out)
