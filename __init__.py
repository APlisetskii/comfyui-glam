import random
from PIL import Image
import numpy as np
import torch

#
# === Старая нода: GlamRandomImage ===
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
        
        # Find highest connected image number
        max_connected = 0
        for i in range(1, 12):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                max_connected = i

        # Always add one empty slot after the last connected image
        if max_connected < 11:
            next_slot = f"image_{max_connected + 1}"
            input_types["required"][next_slot] = ("IMAGE",)

    def process(self, *args, **kwargs):
        seed = 0
        if "seed" in kwargs:
            seed = kwargs["seed"]
            del kwargs["seed"]

        # Filter out None values and collect valid images
        images = []
        for i in range(1, 12):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])

        if not images:
            raise ValueError("At least one image input is required")

        random.seed(seed)
        choice = random.choice(images)
        
        # Update slots for next execution
        self.update_input_types(**kwargs)
        
        return (choice,)

#
# === Новая версия GlamSmoothZoom с "Image.transform" (AFFINE) ===
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
                "interpolation": (["LANCZOS", "BICUBIC", "BILINEAR"], {"default": "LANCZOS"}),
                "easing": (["linear", "ease_in_out", "ease_out"], {"default": "ease_in_out"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "comfyui-glam-nodes"

    # easing-функции
    def ease_in_out(self, t):
        return t * t * (3 - 2 * t)

    def ease_out(self, t):
        return 1 - (1 - t) ** 3

    def process(self, image, zoom_factor, duration, fps, interpolation, easing):
        """
        Анимационный зум вокруг центра:
        - Генерируем N кадров (N=fps*duration),
        - Каждый кадр делается через аффинное преобразование (subpixel),
          что убирает дергания от неоднородного int/round при crop+resize.
        - Возвращаем PyTorch-тензор (N,H,W,C).
        """

        # 1) Преобразуем вход (PyTorch -> NumPy)
        if isinstance(image, torch.Tensor):
            frame_0 = image[0].detach().cpu().numpy()  # (H, W, C)
        else:
            frame_0 = image[0]

        frame_0 = (frame_0 * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_0)

        width, height = pil_image.size
        total_frames = int(fps * duration)

        # 2) Easing-мэппинг
        def apply_easing(raw_t):
            if easing == "ease_in_out":
                return self.ease_in_out(raw_t)
            elif easing == "ease_out":
                return self.ease_out(raw_t)
            return raw_t  # linear

        # 3) Выбираем метод ресемплинга
        from PIL import Image as PILImage
        resample_method = {
            "LANCZOS":  PILImage.Resampling.LANCZOS,
            "BICUBIC":  PILImage.Resampling.BICUBIC,
            "BILINEAR": PILImage.Resampling.BILINEAR
        }[interpolation]

        # Центр (around which we scale)
        cx = width  / 2.0
        cy = height / 2.0

        frames_list = []
        for i in range(total_frames):
            # 4) t от 0 до 1
            if total_frames > 1:
                t = i / (total_frames - 1)
            else:
                t = 0.0

            t = apply_easing(t)
            scale = 1.0 + zoom_factor * t

            # 5) Формируем аффинное преобразование:
            #   x_in = scale * x_out + (1 - scale)*cx
            #   y_in = scale * y_out + (1 - scale)*cy
            # => coefficients (a, b, c, d, e, f)
            #    x_in = a*x_out + b*y_out + c
            #    y_in = d*x_out + e*y_out + f
            a = scale
            b = 0.0
            c = (1.0 - scale) * cx
            d = 0.0
            e = scale
            f = (1.0 - scale) * cy

            # 6) Применяем transform (выводим в исходный размер (width x height))
            coeffs = (a, b, c, d, e, f)
            transformed = pil_image.transform(
                (width, height),    # output size
                PILImage.AFFINE,    # mode
                coeffs,             # transform coefficients
                resample=resample_method
            )

            # 7) float32, 0..1
            frame_array = np.array(transformed, dtype=np.float32) / 255.0
            frames_list.append(frame_array)

        # 8) Собираем в (N, H, W, C), затем -> Torch
        frames_np = np.stack(frames_list, axis=0)
        frames_torch = torch.from_numpy(frames_np)

        return (frames_torch,)


#
# === Обновляем словарь маппинга нод ===
#
NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
    "GlamSmoothZoom": GlamSmoothZoom,
}

# Папка для статических файлов (если используете web UI/JS/TS скрипты)
WEB_DIRECTORY = "./js"
