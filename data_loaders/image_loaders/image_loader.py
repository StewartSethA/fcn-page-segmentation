def load_image(image_path):
    raise NotImplementedError("No image loader has been specified.")
    
try:
    from cv2_image_loader import load_image
except ImportError:
    from pil_image_loader import load_image
except ImportError:
    from plt_image_loader import load_image
