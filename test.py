import torch
import torch.fft
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决你之前的报错：强制使用 TkAgg 后端或直接保存
matplotlib.use('TkAgg')

# 1. 读取图片
image_path = 'img.png' # 请确保图片路径正确
img_pil = Image.open(image_path).convert('L')
img_tensor = torch.from_numpy(np.array(img_pil)).float()

# 2. 正向傅里叶变换
f_transform = torch.fft.fftn(img_tensor)
f_shift = torch.fft.fftshift(f_transform) # 中心化

# --- 在这里你可以进行频域滤波（可选）---
# 例如：如果你把 f_shift 的某些部分设为 0，就能实现去噪或模糊

# 3. 逆傅里叶变换
# 第一步：必须先逆中心化（把低频移回四个角）
f_ishift = torch.fft.ifftshift(f_shift)
# 第二步：执行逆变换
img_back_complex = torch.fft.ifftn(f_ishift)
# 第三步：取实部（由于浮点运算误差，复数中可能残留极小的虚部，取实部即可恢复图像）
img_back = torch.real(img_back_complex)

# 4. 可视化对比
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(img_tensor, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(132)
# 显示幅度谱
magnitude_spectrum = 20 * torch.log(torch.abs(f_shift) + 1)
plt.imshow(magnitude_spectrum.numpy(), cmap='gray')
plt.title('Spectrum')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed (IFFT)')
plt.axis('off')

plt.tight_layout()
plt.show()
