import random
from PIL import Image
import numpy
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
# === Новая продвинутая нода GlamSmoothZoom ===
#
class GlamSmoothZoom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Принимаем ровно одно изображение (image)
        и несколько управляющих параметров.
        """
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

    #
    # Дополнительные easing-функции
    #
    def ease_in_out(self, t):
        """Плавное ускорение-замедление (S-образная кривая)."""
        return t * t * (3 - 2 * t)

    def ease_out(self, t):
        """Резкий старт и плавная остановка."""
        return 1 - (1 - t) ** 3

    def process(self, image, zoom_factor, duration, fps, interpolation, easing):
        """
        Создаёт батч кадров (shape = [количество_кадров, H, W, C]),
        плавно увеличивая масштаб одного входного изображения.
        """
        # Из ComfyUI батч обычно приходит как (batch, height, width, channels).
        # Берём только первый кадр (если batch > 1).
        frame_0 = image[0]  # shape: (height, width, channels)
        frame_0 = (frame_0 * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_0)

        width, height = pil_image.size
        total_frames = int(fps * duration)

        # Определяем метод ресемплинга
        resample_method = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BICUBIC": Image.Resampling.BICUBIC,
            "BILINEAR": Image.Resampling.BILINEAR
        }[interpolation]

        frames = []
        for i in range(total_frames):
            # Нормируем t от 0 до 1
            t = i / max(total_frames - 1, 1)

            # Применяем easing
            if easing == "ease_in_out":
                t = self.ease_in_out(t)
            elif easing == "ease_out":
                t = self.ease_out(t)
            # если linear, оставляем t как есть

            current_scale = 1.0 + zoom_factor * t
            new_w = int(width * current_scale)
            new_h = int(height * current_scale)
            scaled_img = pil_image.resize((new_w, new_h), resample=resample_method)

            # Обрезаем по центру до исходного размера
            left = (scaled_img.width - width) // 2
            top = (scaled_img.height - height) // 2
            right = left + width
            bottom = top + height
            cropped = scaled_img.crop((left, top, right, bottom))

            # Конвертируем в float32 для ComfyUI
            frame_array = np.array(cropped, dtype=np.float32) / 255.0
            frames.append(frame_array)

        # Собираем всё в один numpy-стек (batch)
        frames = np.stack(frames, axis=0)  # (total_frames, H, W, C)
        return (frames,)


#
# === Обновляем словарь маппинга нод ===
#
NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
    "GlamSmoothZoom": GlamSmoothZoom,
}

# Папка для статических файлов (если используете web UI/JS/TS скрипты)
WEB_DIRECTORY = "./js"
