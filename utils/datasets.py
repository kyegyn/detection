import cv2
import numpy as np
from io import BytesIO
from random import choice, random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from torchvision import datasets
import torchvision.transforms.functional as TF
from PIL import Image  # 补充ImageOps
from utils import CONFIGCLASS
# -------------------------- 修复后的辅助函数 --------------------------
def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img = img.astype(np.uint8)  # 强制uint8类型
    img_cv2 = img[:, :, ::-1]  # RGB→BGR
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    if not result:
        raise RuntimeError(f"JPEG压缩失败，质量参数：{compress_val}")
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]  # BGR→RGB

def pil_jpg(img: np.ndarray, compress_val: int):
    img = img.astype(np.uint8)  # 强制uint8类型
    with BytesIO() as out:  # 自动关闭缓冲区
        Image.fromarray(img).save(out, format="jpeg", quality=compress_val)
        out.seek(0)
        img = Image.open(out)
        img = np.array(img)
    return img

def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])

def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        # 高斯模糊（容错处理）
        blur_prob = getattr(cfg, 'blur_prob', 0.5)
        if random() < blur_prob:
            blur_sig = getattr(cfg, 'blur_sig', [0.2, 0.8])
            sig = sample_continuous(blur_sig)
            # 修复：使用副本避免修改原数组
            img[:, :, 0] = gaussian_filter(img[:, :, 0].copy(), sigma=sig)
            img[:, :, 1] = gaussian_filter(img[:, :, 1].copy(), sigma=sig)
            img[:, :, 2] = gaussian_filter(img[:, :, 2].copy(), sigma=sig)

        # JPEG压缩（容错处理）
        jpg_prob = getattr(cfg, 'jpg_prob', 0.5)
        if random() < jpg_prob:
            jpg_method = getattr(cfg, 'jpg_method', ['cv2', 'pil'])
            jpg_qual = getattr(cfg, 'jpg_qual', [50, 95])
            method = sample_discrete(jpg_method)
            qual = sample_discrete(jpg_qual)
            jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}
            if method not in jpeg_dict:
                raise ValueError(f"不支持的JPEG压缩方法：{method}，可选：{list(jpeg_dict.keys())}")
            img = jpeg_dict[method](img, qual)

    return Image.fromarray(img)

# -------------------------- binary_dataset函数（无修改） --------------------------
def binary_dataset(root: str, cfg: CONFIGCLASS):
    identity_transform = transforms.Lambda(lambda img: img)

    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
    else:
        crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform

    return datasets.ImageFolder(
        root,
        transforms.Compose(
            [
                rz_func,
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )
    )

if __name__ == '__main__':
    from utils import cfg  # isort: split
    dataset = binary_dataset(root='train', cfg=cfg)
    print(f"Dataset loaded with {len(dataset)} samples.")
    img, label = dataset[0]  # 取出 (image, label)
    # 如果 transform 已经转为 Tensor，使用 .shape；否则将 PIL 转为 numpy 再查看 shape
    try:
        print('image shape:', img.shape)
    except Exception:
        import numpy as np
        print('image shape:', np.array(img).shape)
    print('label:', label)
