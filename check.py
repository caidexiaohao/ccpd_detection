import os

# 设置数据集路径
img_dir = "datasets/images/train"
label_dir = "datasets/labels/train"

# 支持的图像扩展名
img_exts = ['.jpg', '.png', '.jpeg']

missing_labels = []
empty_labels = []
invalid_lines = []

def is_valid_yolo_line(line):
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        floats = [float(p) for p in parts]
        return all(0.0 <= f <= 1.0 for f in floats[1:])  # class_id 可以超出 0~1，但坐标必须归一化
    except:
        return False

for img_file in os.listdir(img_dir):
    if not any(img_file.endswith(ext) for ext in img_exts):
        continue
    
    label_file = img_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(label_dir, label_file)
    
    if not os.path.exists(label_path):
        missing_labels.append(img_file)
        continue
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            empty_labels.append(img_file)
        else:
            for line in lines:
                if not is_valid_yolo_line(line):
                    invalid_lines.append((img_file, line.strip()))

# 打印结果
print(f"🚫 缺失标签文件数: {len(missing_labels)}")
print(f"⚠️ 空标签文件数: {len(empty_labels)}")
print(f"❌ 非法格式标签行数: {len(invalid_lines)}")

# 可选输出详细内容
if missing_labels:
    print("\nMissing labels:")
    for f in missing_labels[:10]: print(f"  - {f}")
if empty_labels:
    print("\nEmpty labels:")
    for f in empty_labels[:10]: print(f"  - {f}")
if invalid_lines:
    print("\nInvalid format lines:")
    for f, l in invalid_lines[:10]: print(f"  - {f}: '{l}'")
