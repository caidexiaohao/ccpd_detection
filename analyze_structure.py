import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

csv_path = 'structure_error_report.csv'  # 路径可根据你的目录修改
blur_threshold = 100

# 统计容器
total = 0
lengths = Counter()
structure_errors = 0
blurry_images = 0
blur_scores = []
structure_error_images = []

length_diff_counter = Counter()
blur_bins = defaultdict(int)

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1

        true_len = int(row['True Length'])
        pred_len = int(row['Predicted Length'])
        length_diff = pred_len - true_len
        length_diff_counter[length_diff] += 1

        blur = float(row['Blur Score'])
        blur_scores.append(blur)
        if blur < blur_threshold:
            blurry_images += 1
            blur_bins['模糊'] += 1
        else:
            blur_bins['清晰'] += 1

        lengths[pred_len] += 1

        if row['Structure Error'].strip() == 'Yes':
            structure_errors += 1
            structure_error_images.append(row['Image'])

# 输出统计信息
print(f"📊 样本总数: {total}")
print(f"📌 模型预测长度分布:")
for k in sorted(lengths):
    print(f"  - 长度 {k}: {lengths[k]} 个")

print(f"🌫️ 模糊图像数量 (清晰度 < {blur_threshold}): {blurry_images} 张，占比 {blurry_images / total * 100:.2f}%")

print(f"🧠 Structure Error 错误数量: {structure_errors} 个，占比 {structure_errors / total * 100:.2f}%")

print(f"🔍 预测长度与真实长度差异分布:")
for diff in sorted(length_diff_counter):
    print(f"  - 差值 {diff}: {length_diff_counter[diff]} 个")

# 可选画图
plt.figure(figsize=(10, 4))
plt.bar(length_diff_counter.keys(), length_diff_counter.values(), color='orange')
plt.xlabel('预测长度 - 真实长度')
plt.ylabel('样本数量')
plt.title('长度差值分布')
plt.grid(True)
plt.tight_layout()
plt.savefig('length_error_distribution.png')
print("📈 长度差值分布图已保存为 length_error_distribution.png")
