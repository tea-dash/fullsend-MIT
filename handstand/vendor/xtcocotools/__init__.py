# Lightweight stub to satisfy imports in mmpose without compiling xtcocotools.
# Delegates to pycocotools if available.

from .coco import *  # noqa: F401,F403
from .mask import *  # noqa: F401,F403


