import os, gc
from datetime import datetime
import torch
from ultralytics import YOLO

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_resume_model():
    last_path = "runs/detect/train/weights/last.pt"
    if os.path.exists(last_path):
        print("🔁 检测到已有训练记录，将从 last.pt 继续训练")
        return YOLO(last_path), True
    else:
        print("🆕 初始化新模型 yolo11n.pt")
        return YOLO("yolo11n.pt"), False

def train_model(model, resume=False):
    try:
        model.train(
            data='ccpd.yaml',
            epochs=100,
            batch=64,
            device='0',
            resume=resume,
            patience=10,
            save_period=10  # 每隔 10 epoch 保存一次快照
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ 显存不足，自动降低 batch size 到 32")
            release_gpu_memory()
            model.train(data='ccpd.yaml', epochs=100, batch=32, device='0', resume=resume)
        else:
            raise e

def export_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"ccpd_export_{timestamp}.onnx"
    success = model.export(format='onnx', filename=export_name)
    print("✅ 导出成功：" if success else "❌ 导出失败", export_name)

def run_prediction(model):
    result = model.predict(source='test/images', save=True, conf=0.5)
    print("📸 推理完成，检测结果已保存")

# 释放内存 & 加载模型


if __name__ == '__main__':
    release_gpu_memory()
    model, resume_flag = get_resume_model()
    train_model(model, resume_flag)
    export_model(model)
    run_prediction(model)
