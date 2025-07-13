import os
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
             "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = alphabets[:-1] + [str(i) for i in range(10)] + ['O']
input_dir = 'recognition\images'
output_dir = 'recognition\labels'

def parse_plate(filename):
    # 示例文件名
    # "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    parts = filename.split('-')
    char_indices = [int(i) for i in parts[4].split('_')]
    plate = provinces[char_indices[0]] + alphabets[char_indices[1]] + ''.join([ads[i] for i in char_indices[2:]])
    return plate

# 示例调用
for sub_dir in os.listdir(input_dir):
    sub_input_dir = os.path.join(input_dir, sub_dir)
    sub_output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    
    for filename in os.listdir(sub_input_dir):
        if not filename.endswith('.jpg'):
            continue
        plate = parse_plate(filename)
        label_path = os.path.join(sub_output_dir, filename.replace('.jpg', '.txt'))
        
        with open(label_path, 'w',encoding='utf-8') as f:
            f.write(f"{plate}\n")  # 假设类别为0，实际应用中可能需要根据具体情况调整
