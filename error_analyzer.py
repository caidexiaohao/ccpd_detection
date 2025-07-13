import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
from license_plate_dataset import LicensePlateDataset, LicensePlateVocab
from license_plate_model import LicensePlateModel
from torchvision import transforms
import os
import argparse
import csv
import shutil


# 🔤 设置字符表（与你训练时保持一致）
vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
vocab = LicensePlateVocab(vocab_list)
img_size = 224
max_length = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform():
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

def load_model(checkpoint_path):
    model = LicensePlateModel(
        pad_idx=vocab.pad_idx,
        d_model=64,
        nhead_encoder=4,
        nhead_decoder=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        vocab_size=len(vocab.vocab_list),
        max_length=max_length
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

def analyze_errors(model, val_loader, output_dir='error_samples', csv_path='error_report.csv'):
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(exist_ok=True)
    
    report_rows = []
    report_rows = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, labels[:, :-1])      # teacher forcing
            outputs = outputs.permute(0, 2, 1)            # [B, vocab_size, T]
            _, predicted = torch.max(outputs, 1)          # [B, T]
            non_pad_mask = (labels[:, 1:] != vocab.pad_idx)
            exact_match_mask = ((predicted == labels[:, 1:]) | ~non_pad_mask).all(dim=1)  # [B]

            for i in range(images.size(0)):
                if not exact_match_mask[i]:
                    true_seq = vocab.sequence_to_text(labels[i].cpu().numpy())
                    pred_seq = vocab.sequence_to_text(predicted[i].cpu().numpy())
                    filename = f"{batch_idx:03}_{i:02}_{true_seq}_{pred_seq}.png"
                    save_image(images[i], Path(output_dir) / filename)
                    report_rows.append([filename, true_seq, pred_seq])

    # 写入 CSV 报告
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Ground Truth', 'Prediction'])
        writer.writerows(report_rows)

    print(f"✅ 错误样本分析完成，共发现 {len(report_rows)} 个错误。图像保存在 {output_dir}/，详情见 {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="错误样本自动分析器")
    parser.add_argument('--val_folder', type=str, default='./recognition/images/val', help='验证集路径')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_model.pth', help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=64, help='验证批大小')
    parser.add_argument('--label_folder', type=str, default='./recognition/labels/val', help='标签文件夹路径')
    args = parser.parse_args()

    val_dataset = LicensePlateDataset(args.val_folder, args.label_folder, vocab, max_length, get_transform())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = load_model(args.checkpoint_path)

    analyze_errors(model, val_loader)
