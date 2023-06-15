import os
import re
import json
import shutil
import pyewts
import random
import requests
import tensorflow as tf
import urllib.request

from tqdm import tqdm


def shuffle_data(images: list, labels: list) -> tuple[list, list]:
    c = list(zip(images, labels))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


class ImageReader:
    def __init__(self, img_width: int = 2000, img_height: int = 80) -> None:
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.squeeze(img)

        return img


class LabelReader:
    def __init__(self, convert_labels: bool = True) -> None:
        self._converter = pyewts.pyewts()

    def _clean_label(self, label: str) -> str:
        label = label.replace("༌", "་")
        label = re.sub("[\xa0\t\uf8f0]", " ", label)
        label = re.sub("[＠@�■\n\r\u3000\uf038\uf037\ufeff|~0-9|a-z]", "", label)
        label = re.sub("[[(].*?[])]", "", label)

        return label

    def _convert_label(self, label: str) -> str:
        label = self._converter.toWylie(label)
        label = label.replace(" ", "§")  # use $ as placeholder for tseg
        label = label.replace("__", "_")
        return label

    def __call__(self, label_path: str) -> str:
        lbl = open(label_path, encoding="utf-8").readline()
        lbl = self._clean_label(lbl)
        lbl = self._convert_label(lbl)

        return lbl


class TextVectorizer:
    def __init__(
        self,
        charset: list[str],
        sequence_length: int = 500,
        padding_token: int = 0,
        pad_sequences: bool = True,
    ) -> None:
        self.charset = charset
        self.sequence_length = sequence_length
        self.padding_token = padding_token
        self.pad_sequences = pad_sequences

    def __call__(self, item):
        vec_label = [x for x in item]
        vec_label = [self.charset.index(x) for x in vec_label]

        if self.pad_sequences:
            length = tf.shape(vec_label)[0]
            pad_amount = self.sequence_length - length
            vec_label = tf.pad(
                vec_label,
                paddings=[[0, pad_amount]],
                constant_values=self.padding_token,
            )
        return vec_label


class IIIFDownloader:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._current_download_dir = None

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def get_download_dir(self) -> str:
        return self._current_download_dir

    def download(self, manifest_link: str, file_limit: int = 50):
        file_limit = int(file_limit)

        with urllib.request.urlopen(manifest_link) as url:
            data = json.load(url)
            seq = data["sequences"]
            volume_id = seq[0]["@id"].split("bdr:")[1].split("/")[0]

            volume_out = os.path.join(self._output_dir, "Downloaded", volume_id)
            self._current_download_dir = volume_out

            if not os.path.exists(volume_out):
                os.makedirs(volume_out)

            max_images = len(seq[0]["canvases"])

            if max_images > 0:
                if file_limit == 0 or file_limit > max_images or file_limit < 0:
                    file_limit = max_images

                for idx in tqdm(range(file_limit)):
                    img_url = seq[0]["canvases"][idx]["images"][0]["resource"]["@id"]
                    img_name = img_url.split("::")[1].split(".")[0]
                    out_file = f"{volume_out}/{img_name}.jpg"

                    if not os.path.isfile(out_file):
                        res = requests.get(img_url, stream=True)

                        if res.status_code == 200:
                            with open(out_file, "wb") as f:
                                shutil.copyfileobj(res.raw, f)
