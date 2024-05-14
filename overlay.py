import torch

class ImageOverlayNode:
    """
    A custom node for ComfyUI that overlays a cropped image onto a source image at specified coordinates.

    Class Methods:
    --------------
    INPUT_TYPES(cls) -> dict:
        A class method returning a dictionary containing configuration for input fields.

    Attributes:
    -----------
    RETURN_TYPES (tuple): 
        Specifies the types of each element in the output tuple.
    FUNCTION (str): 
        The name of the entry-point method.
    CATEGORY (str): 
        Specifies the category the node should appear in the UI.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Returns a dictionary containing configuration for input fields.

        Returns:
        --------
        dict:
            A dictionary with keys specifying input fields and values specifying their types and default values.
        """
        return {
            "required": {
                "source_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "x_coordinate": ("INT", {"default": 0}),
                "y_coordinate": ("INT", {"default": 0})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_images"
    CATEGORY = "Custom"

    def overlay_images(self, source_image, cropped_image, x_coordinate, y_coordinate):
        """
        Overlays a cropped image onto a source image at specified coordinates.

        Parameters:
        -----------
        source_image : torch.Tensor
            The source image tensor onto which the cropped image will be overlaid.
        cropped_image : torch.Tensor
            The image tensor to be overlaid onto the source image.
        x_coordinate : int
            The x-coordinate of the top-left point where the cropped image will be placed.
        y_coordinate : int
            The y-coordinate of the top-left point where the cropped image will be placed.

        Returns:
        --------
        torch.Tensor:
            The resulting image tensor after overlaying the cropped image onto the source image.
        """
        print(source_image.size())
        print(cropped_image.size())
        # Get the dimensions of the cropped image
        cropped_height, cropped_width = cropped_image.shape[1:3]

        # Get the dimensions of the source image
        source_height, source_width = source_image.shape[1:3]

        # Compute new dimensions if necessary
        new_height = max(source_height, y_coordinate + cropped_height)
        new_width = max(source_width, x_coordinate + cropped_width)

        # Extend the source image with padding if necessary
        if y_coordinate < 0:
            pad_size = abs(y_coordinate)
            pad = torch.ones((source_image.shape[0], pad_size, source_image.shape[2], source_image.shape[3]), dtype=source_image.dtype) * 255  # White padding
            source_image = torch.cat((pad, source_image), dim=1)
            y_coordinate = 0

        # Compute the end coordinates for placing the cropped image
        end_y = min(y_coordinate + cropped_height, new_height)
        end_x = min(x_coordinate + cropped_width, new_width)

        # Compute the start coordinates for placing the cropped image
        start_y = max(y_coordinate, 0)
        start_x = max(x_coordinate, 0)

        # Place the cropped image onto the source image at the specified coordinates
        source_image[:, start_y:end_y, start_x:end_x, :] = cropped_image[:, max(-y_coordinate, 0):max(-y_coordinate, 0) + (end_y - start_y), max(-x_coordinate, 0):max(-x_coordinate, 0) + (end_x - start_x), :]
        print(source_image.size())
        return (source_image,)
    
# Dictionary to map node classes to their names
NODE_CLASS_MAPPINGS = {
    "ImageOverlayNode": ImageOverlayNode
}

# Dictionary to map node names to their friendly display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOverlayNode": "Image Overlay Node"
}
