import random


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
    CATEGORY = "string-tools"

    def get_input_types(self):
        if self.input_types is None:
            self.input_types = self.INPUT_TYPES()
        return self.input_types

    def update_input_types(self, **kwargs):
        input_types = self.get_input_types()
        
        # Find highest connected image number
        max_connected = 1
        for i in range(1, 12):
            if f"image_{i}" in kwargs and kwargs[f"image_{i}"] is not None:
                max_connected = i

        # Add next slot if needed
        if max_connected < 11 and f"image_{max_connected + 1}" not in input_types["required"]:
            input_types["required"][f"image_{max_connected + 1}"] = ("IMAGE",)

    def process(self, *args, **kwargs):
        seed = 0
        if "seed" in kwargs:
            seed = kwargs["seed"]
            del kwargs["seed"]

        self.update_input_types(**kwargs)

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
        return (choice,)


NODE_CLASS_MAPPINGS = {
    "GlamRandomImage": GlamRandomImage,
}
WEB_DIRECTORY = "./js"
