import os
import cv2
import keras
import pyewts
import logging
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import keras.backend as K
from tqdm import tqdm
from glob import glob
from typing import Optional, List
from datetime import datetime
from natsort import natsorted
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils.data_utils import Sequence
from Lib.Models import Easter2
from Lib.Exporter import PageXML, ProdigyLines, TextOutput
from Lib.Utils import shuffle_data, ImageReader, LabelReader, TextVectorizer


class OCRDataset:
    def __init__(
        self,
        directory: str,
        train_test_split: float = 0.8,
        batch_size: int = 32,
        charset: Optional[List[str]] = None,
    ) -> None:
        self._directory = directory
        self._ds_images = []
        self._ds_labels = []
        self._train_idx = []
        self._val_idx = []
        self._test_idx = []
        self._charset = charset
        self._converter = pyewts.pyewts()
        self._label_reader = LabelReader()
        self._batch_size = batch_size
        self._train_test_split = train_test_split
        self._time_stamp = datetime.now()
        self.output_dir = os.path.join(
            self._directory,
            f"Output_{self._time_stamp.year}_{self._time_stamp.month}_{self._time_stamp.day}_{self._time_stamp.hour}_{self._time_stamp.minute}",
        )

        self._init()

    def _init(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ds_images = natsorted(glob(f"{self._directory}/lines/*.jpg"))
        ds_labels = natsorted(glob(f"{self._directory}/transcriptions/*.txt"))
        logging.info(f"Total Images: {len(ds_images)}, Total Labels: {len(ds_labels)}")

        image_names = list(map(self._get_basename, ds_images))
        label_names = list(map(self._get_basename, ds_labels))

        common_list = list(set(image_names) & set(label_names))

        images = list(map(self._map_img_dir, common_list))
        labels = list(map(self._map_label_dir, common_list))

        self._ds_images, self._ds_labels = self._shuffle(images, labels)

        (self._train_idx, self._val_idx, self._test_idx) = self._create_sets(
            self._ds_images, self._ds_labels
        )

        logging.info(f"Train Set: {len(self._train_idx)}")
        logging.info(f"Validation Set: {len(self._val_idx)}")
        logging.info(f"Test Set: {len(self._test_idx)}")

        self._save_dataset("train")
        self._save_dataset("val")
        self._save_dataset("test")

    def _get_basename(self, x):
        return os.path.basename(x).split(".")[0]

    def _shuffle(self, x, y):
        return shuffle_data(x, y)

    def _map_img_dir(self, x):
        return f"{self._directory}/lines/{x}.jpg"

    def _map_label_dir(self, x):
        return f"{self._directory}/transcriptions/{x}.txt"

    def _get_label_set(self, x):
        return set(x)

    def _create_sets(
        self, images: list[str], labels: list[str]
    ) -> tuple[list[int], list[int], list[int]]:
        max_batches = (
            len(images) - (len(images) % self._batch_size)
        ) // self._batch_size

        train_batches = int(max_batches * self._train_test_split)
        val_batches = (max_batches - train_batches) // 2

        train_idx = [int(x) for x in range(train_batches * self._batch_size)]
        val_idx = [
            int(x)
            for x in range(
                (train_batches * self._batch_size),
                (train_batches * self._batch_size + val_batches * self._batch_size),
            )
        ]
        test_idx = [
            int(x)
            for x in range(
                (train_batches * self._batch_size + val_batches * self._batch_size),
                (train_batches * self._batch_size + val_batches * self._batch_size * 2),
            )
        ]
        return train_idx, val_idx, test_idx

    def _save_file(self, outfile: str, entries: list[str]) -> None:
        with open(outfile, "w") as f:
            for entry in entries:
                f.write(f"{entry}\n")

    def _save_dataset(self, split: str):
        imgs_outfile = f"{self.output_dir}/{split}_imgs.txt"
        lbl_outfile = f"{self.output_dir}/{split}_lbls.txt"

        if split == "train":
            idx = self._train_idx
        elif split == "val":
            idx = self._val_idx
        elif split == "test":
            idx = self._test_idx

        else:
            logging.warning(f"{self.__class__.__name__}: invalid split provided, skipping saving dataset")
            return

        imgs = [self._ds_images[x] for x in idx]
        lbls = [self._ds_labels[x] for x in idx]

        self._save_file(imgs_outfile, imgs)
        self._save_file(lbl_outfile, lbls)

    def build_charset(self) -> list[str]:
        labels = [self._label_reader(x) for x in self._ds_labels]
        label_sets = list(map(self._get_label_set, labels))
        flattened_labels = [x for xs in label_sets for x in xs]
        charset = set(flattened_labels)
        charset = sorted(charset)
        charset.append("[BLK]")
        charset.insert(0, "[UNK]")
        self._charset = charset

        return charset

    def get_train_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._train_idx]
        labels = [self._ds_labels[x] for x in self._train_idx]
        return images, labels

    def get_val_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._val_idx]
        labels = [self._ds_labels[x] for x in self._val_idx]
        return images, labels

    def get_test_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._test_idx]
        labels = [self._ds_labels[x] for x in self._test_idx]

        return images, labels

    def get_charset(self) -> list[str]:
        if self._charset is None:
            logging.info(f"Charset has not been built, building charset..")
            self.build_charset()

        return self._charset


class OCRDataLoader(Sequence):
    def __init__(self, images: list[str], labels: list[str], batch_size: int, charset: list[str]):
        self._images = images
        self._labels = labels
        self._charset = (charset,) # for some reason, keras turns this list into a tuple..
        self.img_width: int = 2000
        self.img_height: int = 80
        self._batch_size = batch_size
        self._max_output_length = 500
        self._image_reader = ImageReader(
            img_width=self.img_width, img_height=self.img_height
        )
        self._label_reader = LabelReader()
        self._vectorizer = TextVectorizer(charset=self._charset[0])

    def get_charset(self):
        return self._charset

    def __len__(self):
        return int(np.ceil(len(self._images) / float(self._batch_size)))

    def __getitem__(self, idx):
        # idx = random.randint(0, len(self.image_filenames) // 32)
        image_batch = self._images[idx * self._batch_size : (idx + 1) * self._batch_size]
        label_batch = self._labels[idx * self._batch_size : (idx + 1) * self._batch_size]

        gtTexts = np.ones([self._batch_size, self._max_output_length])
        input_length = np.ones((self._batch_size, 1)) * self._max_output_length
        label_length = np.zeros((self._batch_size, 1))
        imgs = np.ones([self._batch_size, self.img_width, self.img_height])

        for idx in range(0, len(image_batch)):
            imgs[idx] = self._image_reader(image_batch[idx])
            lbl = self._label_reader(label_batch[idx])
            gtTexts[idx] = self._vectorizer(lbl)
            label_length[idx] = len(lbl)
            input_length[idx] = self._max_output_length

        inputs = {
            "the_input": imgs,
            "the_labels": gtTexts,
            "input_length": input_length,
            "label_length": label_length,
        }

        outputs = {"ctc": np.zeros([self._batch_size])}

        return inputs, outputs


class OCRTraining:
    def __init__(
        self,
        model: Model,
        train_loader: OCRDataLoader,
        val_loader,
        test_loader,
        epochs: int = 30,
        checkpoint_path: str = "",
    ) -> None:
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self._batch_size = train_loader._batch_size

        tf.keras.backend.clear_session()

        self.callbacks = [
            ModelCheckpoint(
                filepath=f"{self.checkpoint_path}/model.hdf5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=3, min_lr=1e-8, verbose=1
            ),
            EarlyStopping(patience=5),
        ]

        self.train_steps = len(train_loader) // self._batch_size
        self.val_steps = len(val_loader) // self._batch_size

        self.train_history = None

    def train(self):
        history = self.model.fit(
            self.train_loader,
            epochs=self.epochs,
            validation_data=self.val_loader,
            shuffle=True,
            callbacks=self.callbacks,
        )

        self.train_history = history


class LineDetector:
    def __init__(
        self,
        onnx_model_file: str,
        input_width: int = 1024,
        input_height: int = 192,
        execution_providers: list[str] = ["CPUExecutionProvider"],
    ) -> None:
        self._input_width = input_width
        self._input_height = input_height
        self._onnx_model_file = onnx_model_file
        self.execution_providers = execution_providers  # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self._line_session = None
        self.can_run = False

        self._init()

    def _init(self) -> None:
        if self._onnx_model_file is not None:
            try:
                self._line_session = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
            except Exception as error:
                logging.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        logging.info(f"{self.__class__.__name__} initialized successfully: {self.can_run}")

    def _prepare_line_img(self, img: np.array) -> np.array:
        """
        preprocesses the input image for the line model
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
        img = clahe.apply(img)
        img = cv2.resize(img, (self._input_width, self._input_height))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        return img

    def _get_page_contours(self, predicted_mask: np.array) -> tuple[str, list]:
        predicted_mask = predicted_mask.astype(np.uint8)
        x, y, w, h = cv2.boundingRect(predicted_mask)
        text_region_bbox = f"{x},{y} {x + w},{y} {x + w},{y + h} {x},{y + h}"
        contours, _ = cv2.findContours(
            predicted_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [x for x in contours if cv2.contourArea(x) > 2000]

        contour_dict = {}

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_point = [int(x + w / 2), int(y + h / 2)]

            contour_dict[center_point[1]] = contour

        sorted_contours = {key: contour_dict[key] for key in sorted(contour_dict)}
        sorted_contours = [v for v in sorted_contours.values()]

        return text_region_bbox, sorted_contours

    def _get_line_images(self, image: np.array, line_boxes):
        """
        Input: image of size of the predicted mask return by the line detection
        line_boxes: returned line countours by cv.countours()
        """
        line_images = []

        for idx in range(len(line_boxes)):
            image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            cv2.drawContours(
                image_mask,
                [line_boxes[idx]],
                contourIdx=-1,
                color=(255, 255, 255),
                thickness=-1,
            )
            dilate_k = np.ones((8, 8), dtype=np.uint8)
            kernel_iterations = 6
            image_mask = cv2.dilate(image_mask, dilate_k, iterations=kernel_iterations)

            image_masked = cv2.bitwise_and(image, image, mask=image_mask)

            cropped_img = np.delete(
                image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
            )
            cropped_img = np.delete(
                cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
            )

            line_images.append(cropped_img)

        return line_images

    def run(self, image: np.array) -> tuple[list, np.array, str, list]:
        if self.can_run:
            prepared_img = self._prepare_line_img(image)
            ortvalue = ort.OrtValue.ortvalue_from_numpy(prepared_img)
            results = self._line_session.run_with_ort_values(
                ["conv2d_22"], {"input_1": ortvalue}
            )
            prediction = results[0].numpy()
            prediction = np.squeeze(prediction)
            prediction = np.where(prediction > 0.6, 1.0, 0)
            prediction *= 255
            prediction = prediction.astype(np.uint8)
            pred_mask = cv2.resize(prediction, (image.shape[1], image.shape[0]))
            text_bbox, line_boxes = self._get_page_contours(pred_mask)
            line_images = self._get_line_images(image, line_boxes)

            return line_images, pred_mask, text_bbox, line_boxes
        else:
            logging.error("Error: Line Detector not properly initialzed.")


class OCRInference:
    def __init__(self, model_weights: str, model_characters: list[str]) -> None:
        self._input_width = 2000
        self._input_height = 80
        self._model_weights = model_weights
        self._model_characters = model_characters

        tmp_model = Easter2(self._model_weights, vocab_size=len(self._model_characters))
        self._model = Model(
            tmp_model.get_layer("the_input").input, tmp_model.get_layer("Final").output
        )

    def _prepare_image(self, image: np.array, add_batch_dim: bool = True) -> tf.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize_with_pad(image, self._input_height, self._input_width)
        image = image / 255.0
        image = tf.transpose(image, perm=[1, 0, 2])
        # image = tf.squeeze(image)
        if add_batch_dim:
            image = tf.expand_dims(image, axis=0)  # BxWxHxC

        return image

    def _decode_prediction(self, prediction: np.array):
        if len(prediction.shape) == 2:
            prediction = np.expand_dims(prediction, axis=0)
        out = K.get_value(
            K.ctc_decode(
                prediction,
                input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                greedy=True,
            )[0][0]
        )[0]

        filtered_out = np.delete(out, np.where(out == -1))
        filtered_list = list(filtered_out)
        decoded = "".join(self._model_characters[x] for x in filtered_list)
        decoded = decoded.replace("§", " ")  # replace tseg placeholder

        return decoded

    def run(self, image: np.array, verbose_mode: int = 0) -> tuple[str, np.array]:
        """
        for available verbose modes for keras.models.Model , see: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        """
        image = self._prepare_image(image)
        prediction = self._model.predict(image, verbose=verbose_mode)
        results = self._decode_prediction(prediction)

        return results, prediction

    def run_batched(self, line_images: list) -> list[str]:
        """
        Inputs:
            - expects images to be of shape BxWxHxC, e.g. if you want to run a prediction
            on a whole page with 6 lines, your batch will be of shape (6, 2000, 80, 1)

        """
        line_images = [self._prepare_image(x, add_batch_dim=False) for x in line_images]
        line_images = np.array(line_images)
        prediction = self._model.predict_on_batch(line_images)

        decoded_lines = []
        for pred in prediction:
            decoded_lines.append(self._decode_prediction(pred))

        return decoded_lines


class InferencePipeline:
    def __init__(
        self, line_model_path: str, ocr_model_weights: str, model_characters: list[str]
    ) -> None:
        self._line_model_path = line_model_path
        self._ocr_model_weights = ocr_model_weights
        self._model_characters = model_characters
        self._wylie_converter = pyewts.pyewts()
        self._ocr_model = None
        self._line_detector = None

        self._can_run = False
        self._ocr_model_type = None  # TODO: change to .onnx, or check for .hdf5/.onnx
        self.exporters = []

        self._init()

    def _init(self):
        tf.keras.backend.clear_session()

        if self._line_model_path != "":
            self._line_detector = LineDetector(self._line_model_path)

        if self._ocr_model_weights != "":
            self._ocr_model = OCRInference(
                model_weights=self._ocr_model_weights,
                model_characters=self._model_characters,
            )

        self._can_run = True

        if self._can_run:
            logging.info(f"{self.__class__.__name__}: Ready")
        else:
            logging.error("Failed to setup pipeline")

    def _save_prediction(
        self, image_name, original_image, predicted_mask, out_path, alpha=0.6
    ):
        original_image = original_image.astype(np.float32)
        predicted_mask = predicted_mask.astype(np.float32)
        predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)
        cv2.addWeighted(
            predicted_mask, alpha, original_image, 1 - alpha, 0, original_image
        )
        target_path = os.path.join(out_path, f"{image_name}.jpg")
        cv2.imwrite(target_path, original_image)

    def run(
        self,
        image_dir: str,
        output_formats: list[str],
        mode: str = "batched",
        output_encoding: str = "unicode",
        write_control_image: bool = False,
    ):
        dir_name = os.path.basename(image_dir)
        images = natsorted(glob(f"{image_dir}/*.jpg"))
        line_prediction_out = os.path.join(image_dir, "line_predictions")

        if len(images) == 0:
            logging.error("No images found in directory")
            return

        if write_control_image:
            if not os.path.exists(line_prediction_out):
                os.makedirs(line_prediction_out)

        if "xml" in output_formats:
            xml_exporter = PageXML(image_dir=image_dir)
            self.exporters.append(xml_exporter)

        if "text" in output_formats:
            text_exporter = TextOutput(image_dir=image_dir)
            self.exporters.append(text_exporter)

        if "prodigy" in output_formats:
            prodigy_line_exporter = ProdigyLines(image_dir=image_dir)
            self.exporters.append(prodigy_line_exporter)

        for idx in tqdm(range(len(images)), desc=dir_name):
            image = cv2.imread(images[idx], 1)
            image_name = os.path.basename(images[idx]).split(".")[0]

            predicted_lines, pred_image, text_bbox, line_boxes = self._line_detector.run(
                image
            )

            if len(predicted_lines) == 0:
                continue

            if write_control_image:
                self._save_prediction(image_name, image, pred_image, line_prediction_out)

            if mode == "batched":
                predicted_text_lines = self._ocr_model.run_batched(predicted_lines)

                if output_encoding == "unicode":
                    predicted_text_lines = [
                        self._wylie_converter.toUnicode(x) for x in predicted_text_lines
                    ]

            else:
                predicted_text_lines = []
                for line in predicted_lines:
                    text = self._ocr_model.run(line)

                    if output_encoding == "unicode":
                        text = self._wylie_converter.toUnicode(text)
                    predicted_text_lines.append(text)

            if len(self.exporters) > 0:
                for exporter in self.exporters:
                    exporter.export(
                        image,
                        image_dir,
                        image_name,
                        text_bbox,
                        line_boxes,
                        predicted_lines,
                        predicted_text_lines,
                    )
