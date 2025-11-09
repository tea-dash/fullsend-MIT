try:
    from pycocotools.coco import *  # type: ignore  # noqa: F401,F403
except Exception as e:
    raise ImportError("pycocotools is required for this xtcocotools stub") from e


