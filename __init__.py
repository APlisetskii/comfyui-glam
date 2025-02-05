import random
from PIL import Image
import numpy


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


# === Новая нода: GlamSmoothZoom ===
class GlamSmoothZoom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Указываем, что на вход требуется изображение
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    # Возвращаем батч (список) изображений
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "comfyui-glam-nodes"

    def process(self, image):
        # Получаем первый кадр из батча изображений
        image = image[0]
        
        # Конвертируем тензор в PIL Image для обработки
        image = Image.fromarray((image * 255).astype('uint8'))

        width, height = image.size
        fps = 30
        duration = 5  # в секундах
        total_frames = fps * duration  # 150 кадров
        final_scale = 1.15  # итоговый масштаб (15% зум)
        frames = []

        for frame in range(total_frames):
            # Линейная интерполяция от 1.0 до 1.15
            t = frame / (total_frames - 1)
            scale = 1.0 + t * (final_scale - 1.0)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Масштабируем изображение
            scaled = image.resize((new_width, new_height), resample=Image.LANCZOS)
            # Центрируем и обрезаем до исходного размера
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            frame_img = scaled.crop((left, top, right, bottom))
            frames.append(frame_img)
            
        # Конвертируем обратно в формат ComfyUI
        frames = [numpy.array(frame).astype(numpy.float32) / 255.0 for frame in frames]
        frames = numpy.stack(frames)
        
        return (frames,)


# === Обновление маппинга нод ===
NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
    "GlamSmoothZoom": GlamSmoothZoom,
}
WEB_DIRECTORY = "./js"
