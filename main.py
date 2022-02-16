import codecs
import pytesseract
import re
import os
import sys
import cv2
import json
import numpy as np
import tempfile
import pathlib

from PIL import Image, ImageEnhance
from bs4 import BeautifulSoup
# from urllib.request import urlretrieve
# from pytesseract.pytesseract import image_to_osd
from functions import pretty_print
from wafer_map_excel_ver2 import wafer_map_excel

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_config_file = current_dir.replace("\\main\\dist", "") + '\\config.json'

print(current_dir)
print(path_to_config_file)

f = open(path_to_config_file)
config_json = json.load(f)

# Path to pytesseract files
pytesseract.pytesseract.tesseract_cmd = '.\\pytesseract\\tesseract.exe'

# Path from the config.json file
path_to_html = config_json['html_file_directory']
images_directory = config_json['images_directory']

# # Open and parse the .html file
# file = codecs.open(path_to_html, 'r', 'utf-8')
# soup = BeautifulSoup(file.read(), 'html.parser')

def image_smoothening(img):

    BINARY_THREHOLD = 180

    _, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name: str):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image

def set_image_dpi(img: Image):
    IMAGE_SIZE = 1800

    # img = Image.open(file_path)
    length_x, width_y = img.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = img.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def image_preprocessing_method_1(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255,
                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255,
                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.threshold(cv2.medianBlur(img, 3), 0, 255,
                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                          31, 2)

    cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                          31, 2)

    cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                          31, 2)

    return img

# Similar to method 1, but just lesser processing steps
def image_preprocessing_method_2(img: Image):
    # img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    return img

def image_preprocessing_method_3(img: Image):

    # img = cv2.imread(img_path)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gry, (3, 3), 0)
    thr = cv2.threshold(blr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    (h_thr, w_thr) = thr.shape[:2]
    s_idx = 0
    e_idx = int(h_thr / 2)

    for _ in range(0, 2):
        crp = thr[s_idx:e_idx, 0:w_thr]
        (h_crp, w_crp) = crp.shape[:2]
        crp = cv2.resize(crp, (w_crp * 2, h_crp * 2))
        crp = cv2.erode(crp, None, iterations=1)
        s_idx = e_idx
        e_idx = s_idx + int(h_thr / 2)

    return crp

def image_preprocessing_method_4(img: Image):
    # img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    result = 255 - opening
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result

def image_preprocessing_method_5(img: Image):
    # img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove shadows, cf. https://stackoverflow.com/a/44752405/11089932
    dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img,
                             None,
                             alpha=0,
                             beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8UC1)

    # Threshold using Otsu's
    work_img = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]

    return work_img

def image_preprocessing_method_6(img: Image):

    temp_filename = set_image_dpi(img)
    im_new = remove_noise_and_smooth(temp_filename)

    return im_new

def extract_point_and_defect_fraction_from_img(img: Image, file_name: str):

    try:
        # Configurations to make pytesseract more accurate
        custom_config = r'--oem 3 --psm 6'
        txt = pytesseract.image_to_string(img,
                                          lang='eng',
                                          config=custom_config)

        # Process the text to extract the defect fraction and point number
        # Convert occurences of ¢ to e as pytesseract identifies it wrongly sometimes

        txt = re.sub(r'\b100\b|\b200\b|\b300\b|\b400\b', '', txt)
        txt = re.sub('¢', 'e', txt)
        txt = ''.join(txt.split())

        site_match = re.search('(ML|MLRT)\d+..\d+r2', txt)
        site_text = txt[site_match.start():site_match.end()]
        site = re.sub('(ML|MLRT)\d+..', '', site_text).replace('r2', '')

        match = re.search('Defect\wractionis(\de-\d*|\d.\d*)', txt)
        defect_fraction_text = txt[match.start():match.end()]

        defect_fraction = re.sub('Defect\wractionis', '', defect_fraction_text)

        return {'site': site, 'defect_fraction': defect_fraction}

    except AttributeError:
        raise AttributeError(
            f'Failed to identify point or defect fraction for {file_name}')

# Function definition to extract defect fraction from the image
def preprocess_img_and_extract_point_and_defect_fraction(
        img_path: str, file_name: str):

    try:
        img = image_preprocessing_method_1(img_path)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_2(img)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_3(img)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_4(img)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_5(img)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        # Pass in file path instead
        img = image_preprocessing_method_6(img)
        data = extract_point_and_defect_fraction_from_img(img=img,
                                                          file_name=file_name)

        return data

    except:
        pass

    try:
        img = Image.open(img_path)
        enhancer = ImageEnhance.Sharpness(img)

        factor = 2
        enhanced_img = enhancer.enhance(factor)

        data = extract_point_and_defect_fraction_from_img(img=enhanced_img,
                                                          file_name=file_name)

        return data

    except AttributeError as e:
        raise AttributeError(e)

# # Function definition to download images from the .html file
# def download_images_from_html(folder_dir_to_save='./images'):

#     try:
#         os.makedirs(folder_dir_to_save)
#     except FileExistsError:
#         pass

#     # Iterates through each image found from .html file and save to local folder
#     for index, img in enumerate(soup.find_all('img')):
#         img_src = img['src']

#         # Saves the image file to folder
#         urlretrieve(img_src, f'{folder_dir_to_save}/image_{index}.png')

def main():
    site_defect_fraction_data = []

    try:
        images_dir = images_directory
        image_files = os.listdir(images_dir)

        # download_images_from_html(folder_dir_to_save=images_dir)

        for img_index in range(len(image_files)):
            pretty_print(
                f'Extracting area defect fraction and site number data from image_{img_index}.png...'
            )

            img_dir = f'{images_dir}/image_{img_index}.png'

            try:
                data = preprocess_img_and_extract_point_and_defect_fraction(
                    img_path=img_dir, file_name=f'image_{img_index}.png')

                site_defect_fraction_data.append(data)

            except Exception as e:
                site_defect_fraction_data.append(None)
                print(e)

    except KeyboardInterrupt:
        sys.exit()

    wafer_map_excel(site_defect_fraction_data=site_defect_fraction_data)

# main()


# Mock data would be retrieved from `data.json` -- the part of reading the data from the images would be skipped
def mock_data():

    path_to_data = current_dir.replace("\main", "") + '\\data.json'
    f = open(path_to_data)
    data_json = json.load(f)

    wafer_map_excel(site_defect_fraction_data=data_json)

    # Artificially pauses
    input('Press enter to exit the program...')


mock_data()
