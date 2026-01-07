import torch
from PIL import Image
from models.tsf_net import TSFNet
from data.transforms import get_val_transform
import torch.nn.functional as F

class AIImageDetector:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.transform = get_val_transform(224) # 必须和训练时验证集一致

        # 初始化配置 (这里最好把 config 独立出来读取)
        config = {
            'clip_model': "openai/clip-vit-base-patch32",
            'embed_dim': 256,
            'device': device
            # ... 其他参数 ...
        }

        print("⏳ Loading model for inference...")
        self.model = TSFNet(config).to(device)

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("✅ Model loaded successfully!")

    def predict(self, image_path):
        """
        单张图片预测
        Returns:
            dict: {
                'label': 'Real' or 'Fake',
                'score': float (0~1, probability of being Fake),
                'latency': float (可选)
            }
        """
        # 1. 预处理
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {'error': f"Image load failed: {str(e)}"}

        img_tensor = self.transform(image).unsqueeze(0).to(self.device) # Add batch dim [1, C, H, W]

        # 2. 推理
        with torch.no_grad():
            # 假设 model 需要处理 clip 逻辑
            logits, _, _ = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)

            fake_prob = probs[0, 1].item() # 假设 1 是 Fake
            real_prob = probs[0, 0].item()

        # 3. 格式化输出
        result = "AI-Generated" if fake_prob > 0.5 else "Real"
        confidence = fake_prob if result == "AI-Generated" else real_prob

        return {
            'label': result,
            'confidence': round(confidence, 4),
            'details': {
                'fake_prob': round(fake_prob, 4),
                'real_prob': round(real_prob, 4)
            }
        }

# --- 测试代码 (类似于 main) ---
if __name__ == "__main__":
    # 模拟 API 调用
    detector = AIImageDetector("checkpoints/best_model.pth")

    test_img = "test_samples/unknown.jpg"
    # 创建一个假文件测试一下
    # result = detector.predict(test_img)
    # print(result)
