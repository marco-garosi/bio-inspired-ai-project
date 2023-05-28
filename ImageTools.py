from PIL import Image, ImageDraw, ImageChops
import numpy as np

class ImageTools():
    """Tools to manipulate images
    """

    def __init__(self, image_path, load=True):
        """
        Construct an `ImageTools` object. The image, whose path is given as
        a parameter, can be loaded during construction by setting `load=True`,
        which is the default value. Otherwise, setting it to false will prevent
        the image from being loaded from the file system. Before accessing the
        image, therefore, it will have to be manually by calling the `load_image()`
        method.
        """

        self.image_path = image_path

        if load:
            self.load_image()
        else:
            self.original_image = None
            self.target_image = None
            self.max_difference = None

    def load_image(self):
        """Load an image from the file system
        
        The image loaded from the file system is stored both in the `original_image`
        and `target_image` properties.
        The maximum difference is also computed, so that it is stored and it is not
        necessary to compute it anymore.
        """

        self.original_image = Image.open(self.image_path).convert('RGB')
        self.target_image = self.original_image
        self.max_difference = self.max_pixel_difference(self.target_image)

    def resize_image(self, new_width):
        """Resize an image to a new width preserving the original aspect ratio
        """

        if not self.original_image:
            raise ValueError('No image was loaded! Please load the image first')

        width, height = self.original_image.size
        new_height = int(height * new_width / width)
        
        self.target_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def draw_solution(self, solution):
        """Render a solution by drawing all the polygons
        
        The polygons are drawn on a canvas whose size is the same as that of the target image,
        so that the target image and the solution can be effectively compared.
        """
        
        image = Image.new('RGB', self.target_image.size)
        canvas = ImageDraw.Draw(image, 'RGBA')
        
        for polygon in solution:
            canvas.polygon(polygon[1:], fill=polygon[0])

        return image
    
    def evaluate(self, solution):
        """
        Evaluate the given solution.
        The solution is a list with the following structure:
        [
            (red, green, blue, alpha),
            polygon_1,
            ...,
            polygon_n
        ]

        The solution is rendered as an image and it is then compared
        with the target.
        """

        if not self.target_image:
            raise ValueError('No image was loaded! Please load the image first')

        this_solution = self.draw_solution(solution)
        difference = ImageChops.difference(self.target_image, this_solution)
        absolute_difference = np.array(difference.getdata()).sum()
        return (float(1 - absolute_difference / self.max_difference),)    
    
    def max_pixel_difference(self, target_image):
        """
        Computes the maximum pixel difference from white canvas
        """

        if not target_image:
            raise ValueError('No image was loaded! Please load the image first')

        white = Image.new('RGB', target_image.size, (255, 255, 255))
        diff = ImageChops.difference(white, target_image)
        maxdiff = np.array(diff.getdata()).sum()
        return maxdiff