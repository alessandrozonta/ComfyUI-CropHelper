import unittest
import torch
from PIL import Image
from overlay import ImageOverlayNode
import numpy as np
from torchvision.utils import save_image

class TestImageOverlayNode(unittest.TestCase):
    def test_overlay_images(self):
        # Load sample source image from disk and convert to tensor
        source_image_path = "imgs/body.jpg"
        source_image_pil = Image.open(source_image_path)
        source_image_tensor = torch.tensor(np.array(source_image_pil)).unsqueeze(0).float()  # Convert to tensor

        # Load sample cropped image from disk and convert to tensor
        cropped_image_path = "imgs/face.png"
        cropped_image_pil = Image.open(cropped_image_path)
        cropped_image_tensor = torch.tensor(np.array(cropped_image_pil)).unsqueeze(0).float()  # Convert to tensor

        # Create an instance of the ImageOverlayNode
        overlay_node = ImageOverlayNode()

        # Test overlay with positive coordinates
        x_coordinate = 600
        y_coordinate = 200
        result_image = overlay_node.overlay_images(source_image_tensor, cropped_image_tensor, x_coordinate, y_coordinate)[0]
        self.assertEqual(result_image.size(),source_image_tensor.size())  # Check resulting image size
        
        # Test overlay with negative y-coordinate (should pad the source image)
        y_coordinate = -200
        result_image = overlay_node.overlay_images(source_image_tensor, cropped_image_tensor, x_coordinate, y_coordinate)[0]
        self.assertEqual(result_image.size(), (1,source_image_tensor.size(1) + abs(y_coordinate),  source_image_tensor.size(2) , source_image_tensor.size(3)))  # Check resulting padded image size

if __name__ == '__main__':
    unittest.main()
