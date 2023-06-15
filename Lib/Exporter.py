import os
import cv2
import json
import jsonlines
import numpy as np
from datetime import datetime
from xml.dom import minidom
import xml.etree.ElementTree as etree



class TextOutput:
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.output_dir = None

        self._init()

    def _init(self):
        output_dir = os.path.join(self.image_dir, "text")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir

    def export(
        self,
        image: np.array,
        image_dir: str,
        image_name: str,
        text_bbox,
        line_boxes: list,
        line_images: list[np.array],
        text_lines: list[str],
    ) -> None:
        out_file = f"{self.output_dir}/{image_name}.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            for line in text_lines:
                f.write(f"{line}\n")


class PageXML:
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.output_dir = None

        self._init()

    def _init(self):
        output_dir = os.path.join(self.image_dir, "page")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir

    def _get_time(self):
        t = datetime.now()
        s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        s = s.split(" ")

        return f"{s[0]}T{s[1]}"

    def _get_text_points(self, contour):
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    def _get_text_line_block(self, coordinate, index, unicode_text):
        text_line = etree.Element(
            "Textline", id="", custom=f"readingOrder {{index:{index};}}"
        )
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords
        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line

    def build_xml_document(
        self,
        image: np.array,
        image_name: str,
        text_region_bbox: tuple,
        line_coordinates: list,
        text_lines: list,
    ):
        root = etree.Element("PcGts")
        root.attrib[
            "xmlns"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib[
            "xsi:schemaLocation"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = self._get_time()

        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = text_region_bbox

        for i in range(0, len(line_coordinates)):
            text_coords = self._get_text_points(line_coordinates[i])
            if text_lines != None:
                if len(text_lines[i]) != 0:
                    text_region.append(
                        self._get_text_line_block(
                            text_coords, i, unicode_text=text_lines[i]
                        )
                    )
                else:
                    text_region.append(
                        self._get_text_line_block(text_coords, i, unicode_text="")
                    )
            else:
                text_region.append(
                    self._get_text_line_block(text_coords, i, unicode_text="")
                )

        xmlparse = minidom.parseString(etree.tostring(root))
        prettyxml = xmlparse.toprettyxml()

        return prettyxml

    def export(
        self,
        image: np.array,
        image_dir: str,
        image_name: str,
        text_bbox: str,
        line_boxes: list[np.array],
        line_images: list[np.array],
        text_lines: list[str],
    ):
        xml = self.build_xml_document(
            image, image_name, text_bbox, line_boxes, text_lines
        )

        with open(f"{self.output_dir}/{image_name}.xml", "w", encoding="utf-8") as f:
            f.write(xml)


class ProdigyLines:
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.output_dir = None
        self.line_image_dir = None
        self._init()

    def _init(self):
        output_dir = os.path.join(self.image_dir, "prodigy_lines")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir

        line_image_dir = os.path.join(self.output_dir, "lines")

        if not os.path.exists(line_image_dir):
            os.makedirs(line_image_dir)

        self.line_image_dir = line_image_dir

    def export(
        self,
        image: np.array,
        image_dir: str,
        image_name: str,
        text_bbox: str,
        line_boxes: list[np.array],
        line_images: list[np.array],
        text_lines: list[str],
    ):
        records = []

        for idx in range(len(text_lines)):
            line_image_out_path = f"{self.line_image_dir}/{image_name}_{idx}.jpg"
            prodigy_record = {
                "id": f"{self.line_image_dir}/{image_name}_{idx}",
                "image_url": line_image_out_path,
                "user_input": text_lines[idx],
            }

            records.append(prodigy_record)

            cv2.imwrite(line_image_out_path, line_images[idx])

        jsonl_path = f"{self.output_dir}/{image_name}.jsonl"

        with jsonlines.open(jsonl_path, mode="w") as writer:
            writer.write_all(records)
