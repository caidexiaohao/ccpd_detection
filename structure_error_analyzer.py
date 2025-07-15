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
import cv2
import numpy as np

# ⚙️ 设置字符表
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

def compute_blur_score(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return round(score, 2)

def load_model(checkpoint_path):
    model = LicensePlateModel(
        pad_idx=vocab.pad_idx,
        d_model=64,  # 或与你模型保持一致
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

def analyze_errors(model, val_loader, output_dir='error_samples2', csv_path='structure_error_report.csv'):
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(exist_ok=True)
    report_rows = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output_logits, length_logits = model(images, labels[:, :-1])
            predicted_indices = torch.argmax(output_logits, dim=2)
            predicted_lengths = torch.argmax(length_logits, dim=1)
            true_lengths = (labels[:, 1:] != vocab.pad_idx).sum(dim=1)

            non_pad_mask = (labels[:, 1:] != vocab.pad_idx)
            exact_match_mask = ((predicted_indices == labels[:, 1:]) | ~non_pad_mask).all(dim=1)

            for i in range(images.size(0)):
                true_seq = vocab.sequence_to_text(labels[i].cpu().numpy())
                raw_pred_seq = vocab.sequence_to_text(predicted_indices[i].cpu().numpy())
                trimmed_pred_seq = vocab.sequence_to_text(predicted_indices[i][:predicted_lengths[i]].cpu().numpy())
                blur_score = compute_blur_score(images[i].cpu())

                length_match = int(predicted_lengths[i].item() == true_lengths[i].item())
                structure_error = int(not exact_match_mask[i] or not length_match)

                filename = f"{batch_idx:03}_{i:02}_{true_seq}_{trimmed_pred_seq}.png"
                save_image(images[i], Path(output_dir) / filename)

                report_rows.append([
                    filename,
                    true_seq,
                    raw_pred_seq,
                    trimmed_pred_seq,
                    true_lengths[i].item(),
                    predicted_lengths[i].item(),
                    blur_score,
                    'Yes' if blur_score < 100 else 'No',
                    'Yes' if structure_error else 'No'
                ])

    # 保存 CSV 报告
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Image',
            'Ground Truth',
            'Raw Prediction',
            'Trimmed Prediction',
            'True Length',
            'Predicted Length',
            'Blur Score',
            'Is Blurry',
            'Structure Error'
        ])
        writer.writerows(report_rows)

    print(f"✅ 结构错误分析完成，共分析 {len(report_rows)} 条。图像保存在 {output_dir}/，详情见 {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="结构错误样本分析器")
    parser.add_argument('--val_folder', type=str, default='./recognition/images/val', help='验证集路径')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints2/last_model.pth', help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=64, help='验证批大小')
    parser.add_argument('--label_folder', type=str, default='./recognition/labels/val', help='标签文件夹路径')
    args = parser.parse_args()

    val_dataset = LicensePlateDataset(args.val_folder, args.label_folder, vocab, max_length, get_transform())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = load_model(args.checkpoint_path)
    analyze_errors(model, val_loader)
