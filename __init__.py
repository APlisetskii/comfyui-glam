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

    def ease_in_out(self, t):
        return t * t * (3 - 2 * t)

    def ease_out(self, t):
        return 1 - (1 - t) ** 3

    def process(self, image, zoom_factor, duration, fps, interpolation, easing):
        """
        Создаёт батч кадров (shape = [количество_кадров, H, W, C]),
        плавно увеличивая масштаб одного входного изображения,
        и возвращает PyTorch-тензор, чтобы другие ноды могли делать .cpu().
        """
        # Если image — это PyTorch-тензор, переводим в NumPy
        if isinstance(image, torch.Tensor):
            frame_0 = image[0].detach().cpu().numpy()  # (height, width, channels)
        else:
            frame_0 = image[0]

        frame_0 = (frame_0 * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_0)

        width, height = pil_image.size
        total_frames = int(fps * duration)

        # Выбираем метод ресемплинга
        resample_method = {
            "LANCZOS": Image.Resampling.LANCZOS,
            "BICUBIC": Image.Resampling.BICUBIC,
            "BILINEAR": Image.Resampling.BILINEAR
        }[interpolation]

        frames = []
        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)
            if easing == "ease_in_out":
                t = self.ease_in_out(t)
            elif easing == "ease_out":
                t = self.ease_out(t)
            # linear — t не меняем

            current_scale = 1.0 + zoom_factor * t
            new_w = int(width * current_scale)
            new_h = int(height * current_scale)
            scaled = pil_image.resize((new_w, new_h), resample=resample_method)

            # Обрезаем по центру
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            right = left + width
            bottom = top + height
            cropped = scaled.crop((left, top, right, bottom))

            # Конвертируем в float32 (0..1)
            frame_array = np.array(cropped, dtype=np.float32) / 255.0
            frames.append(frame_array)

        # Собираем кадры в единый NumPy-массив
        frames = np.stack(frames, axis=0)  # (total_frames, H, W, C)

        # !!! Конвертируем в PyTorch-тензор, чтобы следующая нода могла .cpu()
        frames_torch = torch.from_numpy(frames)  # shape (total_frames, H, W, C)

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
