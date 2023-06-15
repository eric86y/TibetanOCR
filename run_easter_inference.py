import os
import logging
import argparse
from Lib.Utils import IIIFDownloader
from config import model_characters
from Lib.Modules import InferencePipeline


"""
to run the pipeline, use e.g.:
python run_easter_inference.py --input_dir "Output" --iiif_manifest "https://iiifpres.bdrc.io/vo:bdr:I1KG81132/manifest" --line_model "Models\LineModels\khyentse_wangpo_q.onnx" --ocr_model "Models\OCRModels\khyentse_wangpo_easter.hdf5" --export_formats "xml,text"
"""


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--line_model", type=str, required=True)
    parser.add_argument("--ocr_model", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--iiif_manifest", type=str, required=False)
    parser.add_argument("--export_formats", type=str, required=False)

    args = parser.parse_args()

    if os.path.isfile(args.line_model) and os.path.isfile( args.ocr_model):
        if args.iiif_manifest:
            logging.debug("running inference from IIIF manifest..")
            iif_downloader = IIIFDownloader(output_dir=args.input_dir)
            iif_downloader.download(args.iiif_manifest)
            input_dir = iif_downloader.get_download_dir()
        else:
            input_dir = args.input_dir

        pipeline = InferencePipeline(
            line_model_path=args.line_model,
            ocr_model_weights=args.ocr_model,
            model_characters=model_characters,
        )

        if len(args.export_formats) > 0:
            output_formats = args.export_formats.split(",")
            print(output_formats)
            
        pipeline.run(input_dir, output_formats=output_formats)

    else:
        logging.info("No valid model paths were provided..")