import random
from PIL import Image
import numpy as np
import torch

#
# === Старая нода: GlamRandomImage (без изменений) ===
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
        """
        zoom_factor: на сколько увеличить (напр. 0.15 = +15%)
        duration:    длительность анимации (сек)
        fps:         кадров в секунду
        interpolation: ["BICUBIC","BILINEAR"] (исключили LANCZOS)
        easing:      тип плавности
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "zoom_factor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 240, "step": 1}),
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
        "Zoom In" (увеличение) без чёрных рамок:
        Мы используем обратную матрицу масштабирования, чтобы 
        PIL не пытался брать пиксели за пределами исходного изображения.
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

        def apply_easing(raw_t):
            if easing == "ease_in_out":
                return self.ease_in_out(raw_t)
            elif easing == "ease_out":
                return self.ease_out(raw_t)
            return raw_t  # linear

        # 2) Выбираем метод (без LANCZOS)
        from PIL import Image as PILImage
        if interpolation == "BICUBIC":
            resample_method = PILImage.Resampling.BICUBIC
        else:
            resample_method = PILImage.Resampling.BILINEAR

        # Центр кадра (не меняется)
        cx = width / 2.0
        cy = height / 2.0

        frames_list = []
        for i in range(total_frames):
            # 3) t: 0..1
            if total_frames > 1:
                t = i / (total_frames - 1)
            else:
                t = 0.0
            t = apply_easing(t)

            # 4) "zoom in" => масштаб растёт от 1.0 до 1.0+zoom_factor
            final_scale = 1.0 + zoom_factor * t

            # 5) В аффинном transform() мы задаём МАТРИЦУ, которая 
            #    описывает, как искать пиксели исходника 
            #    по координатам выходного изображения.
            # 
            #    Если хотим, чтобы результат "выглядел" зумом,
            #    нужно в transform указать коэффициент: 1 / final_scale
            #    То есть мы берём МЕНЬШУЮ часть оригинала, растягивая её на весь (width, height).
            #
            # Коэффициенты для x_in = a*x_out + b*y_out + c
            #                      y_in = d*x_out + e*y_out + f
            #
            # При "zoom in" вокруг (cx, cy):
            #   a = 1.0/final_scale
            #   e = 1.0/final_scale
            #   c = cx*(1 - 1/final_scale)
            #   f = cy*(1 - 1/final_scale)
            #
            a = 1.0 / final_scale
            b = 0.0
            c = cx * (1.0 - a)  # cx - cx*a
            d = 0.0
            e = 1.0 / final_scale
            f = cy * (1.0 - e)  # cy - cy*e

            coeffs = (a, b, c, d, e, f)

            # 6) Применяем transform -> (width, height)
            transformed = pil_image.transform(
                (width, height),
                PILImage.AFFINE,
                coeffs,
                resample=resample_method
            )

            # 7) float32, 0..1
            frame_array = np.array(transformed, dtype=np.float32) / 255.0
            frames_list.append(frame_array)

        # 8) Сформируем батч (N,H,W,C) -> torch
        frames_np = np.stack(frames_list, axis=0)
        frames_torch = torch.from_numpy(frames_np)
        return (frames_torch,)


NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
    "GlamSmoothZoom": GlamSmoothZoom,
}

WEB_DIRECTORY = "./js"
