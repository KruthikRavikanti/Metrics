import cv2
from PIL import Image
import numpy as np
import regex
import pandas as pd
import easyocr
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

import re
from more_itertools import collapse
import copy
from collections import defaultdict
from sortedcontainers import SortedList
from pathlib import Path
import logging

from graph_parser import GraphParser
from get_labels import GetLabelInformation
from hsv2 import *


logger = logging.getLogger(__name__)

class DataExtractor(GraphParser):
    def __init__(self, reader):
        self.reader = reader

    
    # blob detection
    def blob_detection(self, im, scatter, line):
        cpnts, kpnts = [], []
        
        if scatter:
            detector = cv2.SimpleBlobDetector_create()
        elif line:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByConvexity = True
            params.minConvexity = 0.01
            params.filterByInertia = True
            params.minInertiaRatio = 0.0001
            detector = cv2.SimpleBlobDetector_create(params)
        
        img_clone = im.copy()
        while True:
            kp = detector.detect(img_clone)
            if len(kp) == 0:
                break
            kpnts.append(kp)
            for i in kp:
                x, y = round(i.pt[0]), round(i.pt[1])
                cpnts.append((x, y))
                img_clone[y, x] = (0, 0, 0)
                cv2.circle(img_clone, (x, y), round(i.size / 2), (0, 0, 0), -1)
            width, height, _ = img_clone.shape
            for i in range(width):
                for j in range(height):
                    if (img_clone[i, j] == (0, 0, 0)).all():
                        img_clone[i, j] = (255, 255, 255)

        im_with_keypoints = im.copy()
        for i in kpnts:
            im_with_keypoints = cv2.drawKeypoints(
                im_with_keypoints,
                i,
                np.array([]),
                (0, 0, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
        # cv2.imshow('inter', im_with_keypoints)
        # cv2.waitKey(0)
        return im_with_keypoints, cpnts, kpnts
    
    def get_only_colored_blobs(self, im):
        black_img = remove_black_white(im)
        hsvImage, colors = extract_colors(black_img)
        # print(len(colors))
        set_colors = defaultdict(set)
        for i in colors:
            rgb = (HSV_2_RGB(i))
            # print(rgb)
            set_colors[get_colour_name(rgb)].add(i)
        set_colors = dict(sorted(set_colors.items(), key=lambda x:len(x[1]), reverse=True))
        colors_hsv_range = defaultdict(list)
        for k, v in set_colors.items():
            min_h, min_s, min_v = 10000,10000,100000
            max_h, max_s, max_v = 0,0,0
            # print(k, v)
            for i in v:
                hue, sat, val = i[0], i[1], i[2]
                min_h, max_h = min(min_h, hue), max(max_h, hue)
                min_s, max_s = min(min_s, sat), max(max_s, sat)
                min_v, max_v = min(min_v, val), max(max_v, val)
            colors_hsv_range[k] = [min_h, max_h]+[min_s, max_s]+[min_v, max_v]
        for k, v in colors_hsv_range.items():
            lower_mask = np.array([v[0], v[2], v[4]], dtype=np.int32)
            upper_mask = np.array([v[1], v[3], v[5]], dtype=np.int32)
            mask = cv2.inRange(hsvImage, lower_mask, upper_mask)
            if k=='black':
                mask = cv2.bitwise_not(mask)
                img_to_use = cv2.bitwise_and(im, im, mask=mask)

        for r in range(img_to_use.shape[0]):
            for c in range(img_to_use.shape[1]):
                if (img_to_use[r][c]==[0,0,0]).all():
                    img_to_use[r][c] = (255,255,255)
        return img_to_use

    def get_individual_cp(self, inp_img, scatter, line):
        logger.info("Getting color points...")
        image = inp_img
        og_image = copy.deepcopy(image)
        black_img = remove_black_white(image)
        hsvImage, colors = extract_colors(black_img)
        # print(len(colors))
        set_colors = defaultdict(set)
        for i in colors:
            rgb = (HSV_2_RGB(i))
            # print(rgb)
            set_colors[get_colour_name(rgb)].add(i)
        set_colors = dict(sorted(set_colors.items(), key=lambda x:len(x[1]), reverse=True))
        colors_hsv_range = defaultdict(list)
        for k, v in set_colors.items():
            min_h, min_s, min_v = 10000,10000,100000
            max_h, max_s, max_v = 0,0,0
            # print(k, v)
            for i in v:
                hue, sat, val = i[0], i[1], i[2]
                min_h, max_h = min(min_h, hue), max(max_h, hue)
                min_s, max_s = min(min_s, sat), max(max_s, sat)
                min_v, max_v = min(min_v, val), max(max_v, val)
            colors_hsv_range[k] = [min_h, max_h]+[min_s, max_s]+[min_v, max_v]
        # count number of white pixels for each key(color) in the image and sort them in reverse order
        # to get the ones which are detected correctly by the hsv algorithm
        cnt_white = defaultdict(int)
        for k, v in colors_hsv_range.items():
            lower_mask = np.array([v[0], v[2], v[4]], dtype=np.int32)
            upper_mask = np.array([v[1], v[3], v[5]], dtype=np.int32)
            # print(lower_mask, upper_mask)
            mask = cv2.inRange(hsvImage, lower_mask, upper_mask)
            # uncomment below lines to understand the above comment
            # cv2.imshow('mask_'+k, mask)
            # cv2.waitKey(0)
            cnt_white[k] = np.sum(mask == 255)
        colors_to_use = dict(sorted(cnt_white.items(), key=lambda x:x[1], reverse=True))
        colors_to_use = list(colors_to_use.keys())[1:3]
        res = []
        for k, v in colors_hsv_range.items():
            lower = np.array([v[0], v[2], v[4]], dtype=np.uint8)
            upper = np.array([v[1], v[3], v[5]], dtype=np.uint8)
            mask = cv2.inRange(hsvImage, lower, upper)
            if k in colors_to_use:
                masked_image = cv2.bitwise_and(og_image, og_image, mask=mask)
                res.append((k, masked_image))
                # uncomment below lines to see if the algorithm works as intended
                # cv2.imshow('masked_image_'+k, masked_image)
                cv2.imwrite('masked_image_'+k+'.png', masked_image)
                # cv2.waitKey(0)
        all_cp = defaultdict(list)
        for i in range(len(colors_to_use)):
            ind_img = res[i][1].copy()
            ind_img[np.all(ind_img==(0,0,0), axis=-1)] = (255,255,255)
            _, cp, _ = self.blob_detection(ind_img, scatter, line)
            cp = [tuple(reversed(t)) for t in cp]
            all_cp[res[i][0]].append(cp)
        logger.info("Got %d color points." % len(all_cp))
        return all_cp

    def regenerate_graph(self, csv_file):
        df_input = pd.read_csv(csv_file)
        col_list = df_input.columns
        scatter_points_coordinates = [df_input[col].values.tolist() for col in col_list]
        x_cl, y_cl = scatter_points_coordinates[0], scatter_points_coordinates[1]
        # plt.figure(figsize=(8, 5))
        # plt.title("Regenerated Graph")
        # plt.scatter(x_cl, y_cl)
        # plt.show(block=True)

    #def run(self, img_name, scatter, line):
    def run(self, cfg):
        print("Running data extractor...")
        if not Path(cfg.input).exists():
            raise Exception(f"image does not exist at {cfg.input}")
        image_name = re.split(r"\\|/", cfg.input)[-1].split(".")[0]
        Path(cfg.output_prefix).parent.mkdir(parents=True, exist_ok=True)
        image_name = str(Path(cfg.input).name)
        img = self.read_img(cfg.input)
        pil_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)

        # OCR Reader for legend
        # reader = easyocr.Reader(["en"])
        data = self.reader.readtext(img)

        filtered_data = []
        detected_texts = []
        filtered_data.extend(i for i in data if not regex.search(r"\d", i[1]))
        for bbox, text, _ in filtered_data:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]) - 18, int(tl[1])) #what is the reasoning for using the number 18 here?
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]) - 18, int(bl[1]))
            cv2.rectangle(img, tl, br, (255, 255, 255), -1)
            detected_texts.append(text)

        _, img_bin = self.img_to_binary(img)
        edges = self.detect_edges(img_bin)
        lines = self.line_detection(edges)
        xaxis, yaxis, axis_image = self.find_axes(img, lines)

        img_ra = self.remove_axis_lines(axis_image, img)
        img_with_only_colors = self.get_only_colored_blobs(img)
        _, center_points, _ = self.blob_detection(img_with_only_colors, cfg.is_scatter, cfg.is_line)
        # using ransac regressor to get

        x_tick_coords = self.detect_x_axis_ticks(img_bin, xaxis)
        y_tick_coords = self.detect_y_axis_ticks(img_bin, yaxis)

        self.visualize_ticks(img, x_tick_coords, y_tick_coords)

        # Choose which OCR reader to use:

        predicted_text1, predicted_text_center1 = self.text_recognition_easyocr(
            self.reader, str(cfg.input)
            )
        predicted_text2, predicted_text_center2 = self.text_recognition_keras(str(cfg.input))
        predicted_digits1, predicted_digit_center1 = self.filter_digits_ocr(
            predicted_text1, predicted_text_center1
        )
        predicted_digits2, predicted_digit_center2 = self.filter_digits_ocr(
            predicted_text2, predicted_text_center2
        )

        if len(predicted_digits1) > len(predicted_digits2):
            predicted_digits = predicted_digits1
            predicted_digit_center = predicted_digit_center1
        else:
            predicted_digits = predicted_digits2
            predicted_digit_center = predicted_digit_center2

        x_digit_matches, y_digit_matches = self.match_tick_to_digits(
            predicted_digit_center, x_tick_coords, y_tick_coords
        )
        (
            x_digit_coords,
            y_digit_coords,
            x_tick_coords,
            y_tick_coords,
        ) = self.filter_tick_matches_by_dist(
            x_digit_matches,
            y_digit_matches,
            x_tick_coords,
            y_tick_coords,
            predicted_digits,
        )

        reg_x = self.x_axis_regression(x_tick_coords, x_digit_coords)
        reg_y = self.y_axis_regression(y_tick_coords, y_digit_coords)

        x_coord_global_, y_coord_global_ = map(np.array, zip(*center_points))
        x_coord_global_, y_coord_global_ = x_coord_global_.reshape(
            (-1, 1)
        ), y_coord_global_.reshape((-1, 1))
        x_coord_local, y_coord_local = reg_x.predict(x_coord_global_), reg_y.predict(
            y_coord_global_
        )
        x_coord_local, y_coord_local = (
            np.round(x_coord_local, 1).flatten().tolist(),
            np.round(y_coord_local, 1).flatten().tolist(),
        )
        blob_coordinates = list(zip(x_coord_local, y_coord_local))
        blob_coordinates = sorted(blob_coordinates, key=lambda x:x[0])
        reversed_cp = [tuple(reversed(t)) for t in center_points]

        ## uncomment below lines to see if the points are correctly identified
        ## check the order in which the points appear and change the mapping(of blob coordinates and center points) if required
        ## the mapping below this block works for scatter plots
        # a = np.zeros_like(img)
        # for x, y in reversed_cp:
        #     a[x][y]=(255, 255, 255)
        #     cv2.imshow('a',a)
        #     cv2.waitKey(0)
        
        blob_coordinates_for_mapping = sorted(blob_coordinates, key=lambda x:x[1])
        center_points_blob_mapping = {reversed_cp[i]:blob_coordinates_for_mapping[i] for i in range(len(reversed_cp))}
        all_center_pnts = self.get_individual_cp(img, cfg.is_scatter, cfg.is_line)
        colors_points_mapping = defaultdict(SortedList)
        for k, v in center_points_blob_mapping.items():
            for k1, v1 in all_center_pnts.items():
                if k in v1[0]:
                    colors_points_mapping[k1].add(v)
        
        gli = GetLabelInformation(self.reader)
        textual_information = gli.run(cfg.input)
        text_dict = {'Textual Information':list(collapse(textual_information))}

        # dict_for_df = {key.lower().replace(" ", ""):[] for key in legend_data}
        dict_for_df = defaultdict(list)
        
        for blob_coordinate in blob_coordinates:
            dict_for_df["(x,y)"].append((blob_coordinate[0], blob_coordinate[1]))
        
        colors_points_mapping.update(text_dict)
        dict_for_df.update(colors_points_mapping)
        df_output = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dict_for_df.items()]))
        df_output.fillna('', inplace=True)
        df_output.to_csv(cfg.output_prefix + '.csv', index=False)

        fontScale = (img.shape[1] * img.shape[0]) / (500 * 500)
        for i in range(len(center_points)):
            cv2.putText(
                img,
                str(blob_coordinates[i]),
                center_points[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 0, 0),
                1,
            )

        # plt.imshow(img)
        # cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Result", 1500, 1500)

        # creating hssp file
        self.csv_to_json(
            cfg.output_prefix + ".csv",
            "scatter",
            "simpleproject2.json",
            cfg.output_prefix + ".json",
            image_name,
        )
        self.json_to_hssp(
            cfg.output_prefix + '.json', cfg.output_prefix + '.hssp'
        )

        # re-generating the graph
        self.regenerate_graph(cfg.output_prefix + '.csv')


@hydra.main(config_path='conf', config_name='data_extractor', version_base='1.3.2')
def main(cfg: DictConfig):
    reader = easyocr.Reader(['en']) # uncomment "import easyocr" if you want o trun this script as a standalone
    de = DataExtractor(reader)
    de.run(cfg)

if __name__ == "__main__":
    main()
