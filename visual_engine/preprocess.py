"""Image conditioning. Person A replaces the placeholder with OpenCV Canny."""


def make_canny(image_path: str) -> str:
    """Return a path to a Canny edge map.

    Placeholder: returns the original image so the rest of the pipeline
    can be wired before ControlNet is connected.
    """
    return image_path
