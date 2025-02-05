import random
from PIL import Image
import numpy as np
import torch

#
# === Cтарая нода: GlamRandomImage ===
#
class GlamRandomImage:
    def __init__(self):
        self.input_types = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "image_1": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "comfyui-glam-nodes"

    def get_input_types(self):
        if self.input_types is None:
            self.input_types = self.INPUT_TYPES()
        return self.input_types

    def update_input_types(self, **kwargs):
        input_types = self.get_input_types()
        max_connected = 0
        for i in range(1, 12):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                max_connected = i
        if max_connected < 11:
            next_slot = f"image_{max_connected + 1}"
            input_types["required"][next_slot] = ("IMAGE",)

    def process(self, *args, **kwargs):
        seed = 0
        if "seed" in kwargs:
            seed = kwargs["seed"]
            del kwargs["seed"]

        images = []
        for i in range(1, 12):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])

        if not images:
            raise ValueError("At least one image input is required")

        random.seed(seed)
        choice = random.choice(images)
        self.update_input_types(**kwargs)
        return (choice,)


#
# === GlamSmoothZoom (Zoom In, без LANCZOS) ===
#
class GlamSmoothZoom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "zoom_factor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 240, "step": 1}),
                # Убираем LANCZOS, оставляем только BICUBIC и BILINEAR
                "interpolation": (["BICUBIC", "BILINEAR"], {"default": "BICUBIC"}),
                "easing": (["linear", "ease_in_out", "ease_out"], {"default": "ease_in_out"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "comfyui-glam-nodes"

    def ease_in_out(self, t):
        return t * t * (3 - 2 * t)

    def ease_out(self, t):
        return 1 - (1 - t) ** 3

    def process(self, image, zoom_factor, duration, fps, interpolation, easing):
        """
        "Zoom In": Масштаб растет от 1.0 до 1.0+zoom_factor.
        Используем аффинное преобразование через Image.transform,
        чтобы минимизировать подёргивания (jitter).
        Возвращаем PyTorch-тензор (N,H,W,C).
        """

        # 1) PyTorch->NumPy
        if isinstance(image, torch.Tensor):
            frame_0 = image[0].detach().cpu().numpy()
        else:
            frame_0 = image[0]
        frame_0 = (frame_0 * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_0)

        width, height = pil_image.size
        total_frames = int(fps * duration)

        # 2) Подготавливаем easing
        def apply_easing(raw_t):
            if easing == "ease_in_out":
                return self.ease_in_out(raw_t)
            elif easing == "ease_out":
                return self.ease_out(raw_t)
            return raw_t  # linear

        # 3) resample_method (только BICUBIC и BILINEAR)
        from PIL import Image as PILImage
        if interpolation == "BICUBIC":
            resample_method = PILImage.Resampling.BICUBIC
        else:  # BILINEAR
            resample_method = PILImage.Resampling.BILINEAR

        # Координаты центра
        cx = width / 2.0
        cy = height / 2.0

        frames_list = []
        for i in range(total_frames):
            # t: 0..1
            if total_frames > 1:
                t = i / (total_frames - 1)
            else:
                t = 0.0

            t = apply_easing(t)
            # zoom in: 1.0 -> 1.0+zoom_factor
            scale = 1.0 + zoom_factor * t

            # аффинные коэффициенты
            a = scale
            b = 0.0
            c = (1.0 - scale) * cx
            d = 0.0
            e = scale
            f = (1.0 - scale) * cy

            coeffs = (a, b, c, d, e, f)

            # создаём кадр (width x height)
            transformed = pil_image.transform(
                (width, height),
                PILImage.AFFINE,
                coeffs,
                resample=resample_method
            )

            # переводим в float32 (0..1)
            frame_array = np.array(transformed, dtype=np.float32) / 255.0
            frames_list.append(frame_array)

        # формируем батч (N, H, W, C)
        frames_np = np.stack(frames_list, axis=0)
        frames_torch = torch.from_numpy(frames_np)
        return (frames_torch,)


NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
    "GlamSmoothZoom": GlamSmoothZoom,
}

WEB_DIRECTORY = "./js"
