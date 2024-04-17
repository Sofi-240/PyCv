import numpy as np
import os
import os.path as osp
from PIL import Image
from .._lib.array_api.dtypes import cast
from ..colors import Colors

__all__ = [
    "load_image",
    "save_image",
    "ImageLoader",
    "DEFAULT_DATA_PATH"
]

########################################################################################################################

DEFAULT_DATA_PATH = osp.join(osp.dirname(__file__), 'data')


########################################################################################################################

def _from_path(path: str, dtype=None, frame_num: int | None = None) -> np.ndarray:
    if not osp.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    im = Image.open(path)
    frames = []
    fi = 0
    while 1:
        try:
            im.seek(fi)
        except EOFError:
            break

        if frame_num is not None and frame_num != fi:
            _ = im.getdata()[0]
            fi += 1
            continue
        frame = np.asarray(im, dtype=dtype)
        frames.append(frame)
        fi += 1
        if frame_num is not None:
            break

    if not frames:
        return None
    elif len(frames) > 1:
        return np.array(frames)
    return np.array(frames[0])


def _to_path(im: np.ndarray, path: str, format_: str | None = None, **kwargs) -> None:
    inp = cast(im, np.uint8)
    mode_base = None

    if im.ndim == 3:
        inp = cast(im, np.uint8)
        mode = "RGB" if inp.shape[-1] == 3 else "RGBA"
    else:
        mode = 'L'
        mode_base = 'L'

    buffer = inp.tobytes()

    if inp.ndim == 2:
        im = Image.new(mode_base, inp.T.shape)
        im.frombytes(buffer, 'raw', mode)
    else:
        image_shape = (inp.shape[1], inp.shape[0])
        im = Image.frombytes(mode, image_shape, buffer)
    im.save(path, format=format_, **kwargs)


########################################################################################################################

def load_image(path: str, dtype=None, frame_num: int | None = None, _color_fmt: Colors | str | None = None, *args, **kwargs) -> np.ndarray:
    out = _from_path(path, dtype, frame_num)
    if _color_fmt is None:
        return out
    if isinstance(_color_fmt, Colors):
        if not hasattr(_color_fmt, 'function'):
            raise ValueError('_color_fmt need to be member of Colors got Colors')
        return _color_fmt(out, *args, **kwargs)
    elif isinstance(_color_fmt, str):
        return Colors[_color_fmt.upper()](out, *args, **kwargs)
    raise TypeError('invalid _color_fmt type')


def save_image(image: np.ndarray, path: str, format_: str | None = None) -> None:
    if image.ndim not in (2, 3) or image.ndim == 3 and image.shape[-1] not in (3, 4):
        raise ValueError('invalid image shape')

    image = np.asanyarray(image)
    _to_path(image, path, format_)


########################################################################################################################

class ImageLoader(object):
    def __init__(self, dir_path: str | None = None):
        self._dir_path = None
        self.files = {}
        if dir_path is not None:
            self.adapt_dir(dir_path)

    def __repr__(self):
        return str(self.files)

    def adapt_dir(self, dir_path: str):
        if not osp.isdir(dir_path):
            raise FileNotFoundError(f"No such directory: {dir_path}")
        self._dir_path = dir_path
        self.files = {}
        for file in os.listdir(dir_path):
            f = file.split('.')
            if len(f) == 1:
                continue
            self.files[''.join(f[:-1])] = f[-1]

    def load(self, name: str, dtype=None, frame_num: int | None = None, _color_fmt: Colors | str | None = None, *args,
             **kwargs) -> np.ndarray:
        if name not in self.files:
            raise AttributeError(f'no such file {self._dir_path}')
        path = osp.join(self._dir_path, f'{name}.{self.files[name]}')
        return load_image(path, dtype=dtype, frame_num=frame_num, _color_fmt=_color_fmt, *args, **kwargs)

########################################################################################################################
