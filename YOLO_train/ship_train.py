"""
YOLO11s Fine-tuning Script for Ship Detection
소형 선박 이미지 데이터셋을 사용한 GPU 학습 및 재개 기능 포함
"""
import os
import torch
from ultralytics import YOLO

# GPU 사용 확인
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 데이터셋 경로 설정
DATASET_ROOT = r"D:\ProjectVC\datasets\ships-aerial-images"
# DATASET_ROOT = r"D:\ProjectVC\datasets\MASATI-V2"
DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")


# 데이터셋 검증
def verify_dataset():
    """데이터셋 구조 검증"""
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    for split in ['train', 'valid', 'test']:
        img_path = os.path.join(DATASET_ROOT, split, 'images')
        label_path = os.path.join(DATASET_ROOT, split, 'labels')
        if os.path.exists(img_path):
            img_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✓ {split}/images: {img_count} images")
        else:
            print(f"✗ {split}/images: Not found")
        if os.path.exists(label_path):
            label_count = len([f for f in os.listdir(label_path) if f.endswith('.txt')])
            print(f"✓ {split}/labels: {label_count} labels")
        else:
            print(f"✗ {split}/labels: Not found")
    print("="*60 + "\n")

# 학습 함수
def train_model(resume=False, checkpoint_path=None):
    """
    YOLO11 모델 학습
    Args:
        resume (bool): 이전 학습 재개 여부
        checkpoint_path (str): 재개할 체크포인트 경로 (resume=True일 때)
    """
    
    # 모델 로드
    if resume and checkpoint_path:
        print(f"\n📂 Resuming training from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"\n📂 Loading pretrained YOLO11s model...")
        model = YOLO('yolo11s.pt')
    
    # 학습 설정
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    
    training_args = {
        'data': DATA_YAML,
        'epochs': 100,  # 100 with small ship + additional 100 with MASATI-V2
        'imgsz': 512,  # 이미지 크기 (80~800 범위를 512로 리사이즈)
        'batch': 16,   # GPU 메모리에 따라 조정 (8, 16, 32 등)
        'device': 0,   # GPU 0번 사용 (CPU는 'cpu')
        'workers': 0,  # Windows: 0 권장 (multiprocessing 오류 방지), Linux: 8
        'patience': 10,  # Early stopping patience
        'save': True,
        'save_period': 10,  # 10 epoch마다 체크포인트 저장
        'project': 'runs/detect',
        'name': 'ships-aerial_yolo11s',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # SGD, Adam, AdamW 중 선택
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': True,  # 단일 클래스 학습
        'rect': False,  # rectangular training
        'cos_lr': True,  # cosine learning rate scheduler
        'close_mosaic': 10,  # 마지막 N epoch은 mosaic 비활성화
        'resume': resume,  # 학습 재개
        'amp': True,  # Automatic Mixed Precision (GPU 성능 향상)
        'fraction': 1.0,  # 전체 데이터셋 사용
        'profile': False,
        'freeze': None,  # 레이어 동결 (None, 10, [0,1,2] 등)
        'lr0': 0.01,  # 초기 학습률
        'lrf': 0.01,  # 최종 학습률 (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain
        'dfl': 1.5,  # dfl loss gain
        'plots': True,  # 학습 결과 플롯 저장
        'val': True,  # 학습 중 검증
    }
    
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 학습 시작
    try:
        results = model.train(**training_args)
        
        print("\n" + "="*60)
        print("Training Completed Successfully!")
        print("="*60)
        print(f"Best weights saved to: {model.trainer.best}")
        print(f"Last weights saved to: {model.trainer.last}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        raise

# 모델 평가 함수
def evaluate_model(model_path):
    """학습된 모델 평가"""
    import numpy as np
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    model = YOLO(model_path)
    
    # Validation set 평가
    print("\nValidation Set:")
    metrics = model.val(data=DATA_YAML, split='val')
    
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # Precision과 Recall - 배열을 평균으로 변환
    precision = np.array(metrics.box.p) if hasattr(metrics.box.p, '__iter__') else metrics.box.p
    recall = np.array(metrics.box.r) if hasattr(metrics.box.r, '__iter__') else metrics.box.r
    
    if isinstance(precision, np.ndarray) and precision.size > 0:
        print(f"  Precision: {np.mean(precision):.4f}")
    else:
        print(f"  Precision: {precision:.4f}")
    
    if isinstance(recall, np.ndarray) and recall.size > 0:
        print(f"  Recall:    {np.mean(recall):.4f}")
    else:
        print(f"  Recall:    {recall:.4f}")
    
    # Test set 평가
    print("\nTest Set:")
    test_metrics = model.val(data=DATA_YAML, split='test')
    
    print(f"  mAP50:    {test_metrics.box.map50:.4f}")
    print(f"  mAP50-95: {test_metrics.box.map:.4f}")
    
    test_precision = np.array(test_metrics.box.p) if hasattr(test_metrics.box.p, '__iter__') else test_metrics.box.p
    test_recall = np.array(test_metrics.box.r) if hasattr(test_metrics.box.r, '__iter__') else test_metrics.box.r
    
    if isinstance(test_precision, np.ndarray) and test_precision.size > 0:
        print(f"  Precision: {np.mean(test_precision):.4f}")
    else:
        print(f"  Precision: {test_precision:.4f}")
    
    if isinstance(test_recall, np.ndarray) and test_recall.size > 0:
        print(f"  Recall:    {np.mean(test_recall):.4f}")
    else:
        print(f"  Recall:    {test_recall:.4f}")
    
    print("="*60 + "\n")
    
    return metrics

# # 추론 함수
# def predict_sample(model_path, image_path):
#     """샘플 이미지에 대한 추론"""
#     model = YOLO(model_path)
#     results = model.predict(
#         source=image_path,
#         save=True,
#         conf=0.25,
#         iou=0.7,
#         device=0
#     )
#     print(f"\nPrediction saved to: {results[0].save_dir}")
#     return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO11s Ship Detection Fine-tuning")
    print("="*60)
    # # 1. 데이터셋 검증
    # verify_dataset()
    # 2. 학습 모드 선택
    print("Select training mode:")
    print("  1. New training (from pretrained yolo11s.pt)")
    print("  2. Resume training (from checkpoint)")
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    if mode == "1":
        # pretrained model위에 새로운 학습(fine tuning)
        results = train_model(resume=False)
        
    elif mode == "2":
        # 학습 재개
        checkpoint = input("Enter checkpoint path (e.g., runs/detect/MASATI-V2_yolo11s/weights/last.pt): ").strip()
        
        if os.path.exists(checkpoint):
            results = train_model(resume=True, checkpoint_path=checkpoint)
        else:
            print(f"✗ Checkpoint not found: {checkpoint}")
            exit(1)
    else:
        print("Invalid mode selected. Exiting...")
        exit(1)
    
    # 3. 학습 완료 후 평가 (선택)
    evaluate = input("\nEvaluate the model? (y/n): ").strip().lower()
    if evaluate == 'y':
        best_model = "runs/detect/ships-aerial_yolo11s/weights/best.pt"
        if os.path.exists(best_model):
            evaluate_model(best_model)
        else:
            print(f"✗ Best model not found at {best_model}")
    
    print("\n" + "="*60)
    print("Process Completed!")
    print("="*60)