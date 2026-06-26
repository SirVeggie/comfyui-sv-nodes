# this file contains nodes 'borrowed' from another projects
BORROWED_CLASS_MAPPINGS = {}
BORROWED_DISPLAY_NAME_MAPPINGS = {}

#-------------------------------------------------------------------------------#

class GetImageSize:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    CATEGORY = "SV Nodes/Logic"

    FUNCTION = 'get_imagesize'

    def get_imagesize(self, image):
        samples = image.movedim(-1,1)
        size_w = samples.shape[3]
        size_h = samples.shape[2]

        return (size_w, size_h,)

BORROWED_CLASS_MAPPINGS["SV-GetImageSize"] = GetImageSize
BORROWED_DISPLAY_NAME_MAPPINGS["SV-GetImageSize"] = "Image Size"