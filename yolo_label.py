import os
import cv2

# 路径设置
img_dir = "datasets/images/test"
label_dir = "datasets/labels/test"
os.makedirs(label_dir, exist_ok=True)

# 支持图像扩展名
img_exts = ['.jpg', '.jpeg', '.png']

def extract_yolo_label(filename, img_w, img_h):
    try:
        fields = filename.split('-')
        points_str = fields[3].split('_')
        points = [tuple(map(int, p.split('&'))) for p in points_str]

        x_vals = [pt[0] for pt in points]
        y_vals = [pt[1] for pt in points]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    except Exception as e:
        print(f"⚠️ 文件解析失败：{filename} —— {e}")
        return None

total = 0
success = 0
for file in os.listdir(img_dir):
    if not any(file.lower().endswith(ext) for ext in img_exts):
        continue
    total += 1

    img_path = os.path.join(img_dir, file)
    label_path = os.path.join(label_dir, file.rsplit('.', 1)[0] + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像：{file}")
        continue

    h, w = img.shape[:2]
    label_line = extract_yolo_label(file, w, h)
    if label_line:
        with open(label_path, 'w') as f:
            f.write(label_line + '\n')
        success += 1

print(f"\n✅ 标签生成完毕：成功 {success} / 总共 {total}")
