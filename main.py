import pytesseract
import re
import os
import sys
import cv2
import json
import numpy as np
import tempfile

from PIL import Image, ImageEnhance
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from pytesseract.pytesseract import image_to_osd

from wafer_map_excel_ver2 import wafer_map_excel

f = open('config.json')
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


def extract_point_and_defect_fraction_from_img(img: Image):

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
        raise AttributeError('Failed to identify point or defect fraction')


# Function definition to extract defect fraction from the image
def preprocess_img_and_extract_point_and_defect_fraction(img_path: str):

    try:
        img = image_preprocessing_method_1(img_path)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_2(img)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_3(img)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_4(img)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        img = image_preprocessing_method_5(img)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        # Pass in file path instead
        img = image_preprocessing_method_6(img)
        data = extract_point_and_defect_fraction_from_img(img)

        return data

    except:
        pass

    try:
        img = Image.open(img_path)
        enhancer = ImageEnhance.Sharpness(img)

        factor = 2
        enhanced_img = enhancer.enhance(factor)

        data = extract_point_and_defect_fraction_from_img(enhanced_img)

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
    # site_defect_fraction_data = []

    # counter = 0

    # try:
    #     images_dir = images_directory
    #     image_files = os.listdir(images_dir)

    #     # download_images_from_html(folder_dir_to_save=images_dir)

    #     # try:

    #     #     test_img_path = './images/image_6.png'
    #     #     print(
    #     #         preprocess_img_and_extract_point_and_defect_fraction(
    #     #             test_img_path))

    #     # except Exception as e:
    #     #     print(e)

    #     for img_index in range(len(image_files)):
    #         print(f'index: {img_index}')
    #         img_dir = f'{images_dir}/image_{img_index}.png'

    #         try:
    #             data = preprocess_img_and_extract_point_and_defect_fraction(
    #                 img_dir)

    #             site_defect_fraction_data.append(data)
    #             print(data)

    #         except Exception as e:
    #             site_defect_fraction_data.append(None)
    #             print(e)
    #             counter += 1

    #     print(counter)

    # except KeyboardInterrupt:
    #     sys.exit()

    site_defect_fraction_data = [{
        'site': '10',
        'defect_fraction': '0.00181010020020'
    }, {
        'site': '11',
        'defect_fraction': '0.00079'
    }, {
        'site': '12',
        'defect_fraction': '0.00213'
    }, {
        'site': '13',
        'defect_fraction': '0.0015'
    }, {
        'site': '14',
        'defect_fraction': '0.00254'
    }, {
        'site': '15',
        'defect_fraction': '0.00214'
    }, {
        'site': '16',
        'defect_fraction': '0.00162'
    }, {
        'site': '17',
        'defect_fraction': '0.0'
    }, {
        'site': '18',
        'defect_fraction': '0.00027190'
    }, {
        'site': '19',
        'defect_fraction': '0.0037'
    }, {
        'site': '1',
        'defect_fraction': '0.1073'
    }, {
        'site': '20',
        'defect_fraction': '0.0'
    }, {
        'site': '21',
        'defect_fraction': '0.00'
    }, {
        'site': '22',
        'defect_fraction': '0.00343'
    }, {
        'site': '23',
        'defect_fraction': '0.00370'
    }, {
        'site': '24',
        'defect_fraction': '0.00051'
    }, {
        'site': '25',
        'defect_fraction': '0.00179'
    }, {
        'site': '26',
        'defect_fraction': '0.0'
    }, {
        'site': '27',
        'defect_fraction': '0.0'
    }, {
        'site': '28',
        'defect_fraction': '0.00012202'
    }, {
        'site': '29',
        'defect_fraction': '0.00023'
    }, {
        'site': '2',
        'defect_fraction': '0.0268'
    }, {
        'site': '30',
        'defect_fraction': '0.00073'
    }, {
        'site': '31',
        'defect_fraction': '0.0'
    }, {
        'site': '32',
        'defect_fraction': '0.0005'
    }, {
        'site': '33',
        'defect_fraction': '0.00028'
    }, {
        'site': '34',
        'defect_fraction': '0.00013'
    }, {
        'site': '35',
        'defect_fraction': '0.0202'
    }, {
        'site': '36',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '38',
        'defect_fraction': '5e-05'
    }, {
        'site': '39',
        'defect_fraction': '0.0'
    }, {
        'site': '3',
        'defect_fraction': '0.03359'
    }, {
        'site': '40',
        'defect_fraction': '7e-0500'
    }, {
        'site': '41',
        'defect_fraction': '5e-05'
    }, {
        'site': '42',
        'defect_fraction': '7e-05'
    }, {
        'site': '43',
        'defect_fraction': '0.010'
    }, None, {
        'site': '45',
        'defect_fraction': '0.007'
    }, None, {
        'site': '47',
        'defect_fraction': '0.01010020020'
    }, {
        'site': '48',
        'defect_fraction': '0.000531010020020'
    }, {
        'site': '49',
        'defect_fraction': '0.01010020020'
    }, {
        'site': '4',
        'defect_fraction': '0.01788'
    }, {
        'site': '5',
        'defect_fraction': '0.03255'
    }, {
        'site': '6',
        'defect_fraction': '0.02961'
    }, {
        'site': '7',
        'defect_fraction': '0.02166'
    }, {
        'site': '8',
        'defect_fraction': '0.019120'
    }, {
        'site': '9',
        'defect_fraction': '0.022040'
    }, {
        'site': '10',
        'defect_fraction': '0.00544'
    }, {
        'site': '11',
        'defect_fraction': '0.00242'
    }, {
        'site': '12',
        'defect_fraction': '0.00309'
    }, {
        'site': '13',
        'defect_fraction': '0.00047'
    }, {
        'site': '14',
        'defect_fraction': '0.0'
    }, {
        'site': '15',
        'defect_fraction': '0.00399'
    }, {
        'site': '16',
        'defect_fraction': '0.004020'
    }, None, None, {
        'site': '19',
        'defect_fraction': '0.00212'
    }, {
        'site': '1',
        'defect_fraction': '0.13719'
    }, {
        'site': '20',
        'defect_fraction': '0.0047'
    }, None, {
        'site': '22',
        'defect_fraction': '0.00149'
    }, None, None, {
        'site': '25',
        'defect_fraction': '0.0015390'
    }, {
        'site': '26',
        'defect_fraction': '0.0'
    }, None, {
        'site': '28',
        'defect_fraction': '0.00102'
    }, {
        'site': '29',
        'defect_fraction': '0.00024'
    }, {
        'site': '2',
        'defect_fraction': '0.03364'
    }, {
        'site': '30',
        'defect_fraction': '0.00386'
    }, {
        'site': '31',
        'defect_fraction': '0.00082'
    }, {
        'site': '32',
        'defect_fraction': '0.00038'
    }, {
        'site': '33',
        'defect_fraction': '0.00129'
    }, {
        'site': '34',
        'defect_fraction': '0.00023'
    }, {
        'site': '35',
        'defect_fraction': '7e-05'
    }, {
        'site': '36',
        'defect_fraction': '5e-05'
    }, {
        'site': '37',
        'defect_fraction': '0.0'
    }, None, {
        'site': '39',
        'defect_fraction': '6e-05'
    }, {
        'site': '3',
        'defect_fraction': '0.02319'
    }, {
        'site': '40',
        'defect_fraction': '6e-05'
    }, {
        'site': '41',
        'defect_fraction': '0.0'
    }, {
        'site': '42',
        'defect_fraction': '0.0001900'
    }, {
        'site': '43',
        'defect_fraction': '0.0'
    }, None, {
        'site': '45',
        'defect_fraction': '5e-05'
    }, {
        'site': '46',
        'defect_fraction': '7e-05'
    }, None, {
        'site': '48',
        'defect_fraction': '0.000561010020020'
    }, None, {
        'site': '4',
        'defect_fraction': '0.03108'
    }, {
        'site': '5',
        'defect_fraction': '0.01968'
    }, {
        'site': '6',
        'defect_fraction': '0.02489'
    }, {
        'site': '7',
        'defect_fraction': '0.034720'
    }, {
        'site': '8',
        'defect_fraction': '0.03156'
    }, {
        'site': '9',
        'defect_fraction': '0.04707'
    }, {
        'site': '10',
        'defect_fraction': '0.0031'
    }, {
        'site': '11',
        'defect_fraction': '0.001612'
    }, {
        'site': '12',
        'defect_fraction': '0.00016'
    }, {
        'site': '13',
        'defect_fraction': '0.00337'
    }, None, {
        'site': '15',
        'defect_fraction': '0.00365'
    }, {
        'site': '16',
        'defect_fraction': '0.00025'
    }, None, None, {
        'site': '15',
        'defect_fraction': '5e-051010020020'
    }, {
        'site': '1',
        'defect_fraction': '0.15397'
    }, {
        'site': '20',
        'defect_fraction': '0.00198'
    }, {
        'site': '21',
        'defect_fraction': '0.00185'
    }, None, {
        'site': '23',
        'defect_fraction': '0.00076'
    }, {
        'site': '24',
        'defect_fraction': '0.002790'
    }, {
        'site': '25',
        'defect_fraction': '0.00063'
    }, {
        'site': '26',
        'defect_fraction': '0.0008'
    }, None, {
        'site': '28',
        'defect_fraction': '0.00074'
    }, {
        'site': '29',
        'defect_fraction': '0.0007'
    }, {
        'site': '2',
        'defect_fraction': '0.02906'
    }, {
        'site': '30',
        'defect_fraction': '0.00211'
    }, {
        'site': '31',
        'defect_fraction': '5e-05'
    }, {
        'site': '32',
        'defect_fraction': '0.0056'
    }, {
        'site': '33',
        'defect_fraction': '0.000370'
    }, {
        'site': '34',
        'defect_fraction': '0.00023'
    }, {
        'site': '35',
        'defect_fraction': '0.0'
    }, None, {
        'site': '37',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '39',
        'defect_fraction': '0.00016'
    }, {
        'site': '3',
        'defect_fraction': '0.02108'
    }, {
        'site': '40',
        'defect_fraction': '0.00017'
    }, None, {
        'site': '42',
        'defect_fraction': '0.00019'
    }, {
        'site': '43',
        'defect_fraction': '0.0'
    }, {
        'site': '44',
        'defect_fraction': '0.00068'
    }, {
        'site': '45',
        'defect_fraction': '0.0'
    }, None, None, None, None, {
        'site': '4',
        'defect_fraction': '0.026090'
    }, {
        'site': '5',
        'defect_fraction': '0.02494'
    }, {
        'site': '6',
        'defect_fraction': '0.02734'
    }, {
        'site': '7',
        'defect_fraction': '0.0288'
    }, {
        'site': '8',
        'defect_fraction': '0.02549'
    }, {
        'site': '9',
        'defect_fraction': '0.02617'
    }, {
        'site': '10',
        'defect_fraction': '0.00059'
    }, {
        'site': '11',
        'defect_fraction': '0.00276'
    }, None, {
        'site': '13',
        'defect_fraction': '0.00251'
    }, {
        'site': '14',
        'defect_fraction': '0.00121'
    }, None, {
        'site': '16',
        'defect_fraction': '0.00022'
    }, None, {
        'site': '18',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '1',
        'defect_fraction': '0.13654'
    }, {
        'site': '20',
        'defect_fraction': '0.00107'
    }, {
        'site': '21',
        'defect_fraction': '0.000710'
    }, {
        'site': '22',
        'defect_fraction': '0.0'
    }, {
        'site': '23',
        'defect_fraction': '0.00228'
    }, {
        'site': '24',
        'defect_fraction': '0.00162'
    }, {
        'site': '25',
        'defect_fraction': '0.00116'
    }, None, None, {
        'site': '28',
        'defect_fraction': '0.002'
    }, {
        'site': '29',
        'defect_fraction': '0.00109'
    }, {
        'site': '2',
        'defect_fraction': '0.03073'
    }, {
        'site': '301',
        'defect_fraction': '0.0030'
    }, {
        'site': '31',
        'defect_fraction': '0.00075'
    }, {
        'site': '32',
        'defect_fraction': '0.001980'
    }, {
        'site': '33',
        'defect_fraction': '0.00161'
    }, None, {
        'site': '35',
        'defect_fraction': '0.000390'
    }, {
        'site': '36',
        'defect_fraction': '0.00011'
    }, {
        'site': '37',
        'defect_fraction': '0.001990'
    }, {
        'site': '38',
        'defect_fraction': '0.00102'
    }, {
        'site': '39',
        'defect_fraction': '0.00036'
    }, {
        'site': '3',
        'defect_fraction': '0.020520'
    }, {
        'site': '40',
        'defect_fraction': '0.00060'
    }, None, {
        'site': '42',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '44',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '46',
        'defect_fraction': '0.00033'
    }, None, {
        'site': '48',
        'defect_fraction': '0.0'
    }, None, {
        'site': '4',
        'defect_fraction': '0.02582'
    }, {
        'site': '5',
        'defect_fraction': '0.02696'
    }, {
        'site': '6',
        'defect_fraction': '0.02806'
    }, {
        'site': '7',
        'defect_fraction': '0.02893'
    }, {
        'site': '8',
        'defect_fraction': '0.04453'
    }, {
        'site': '9',
        'defect_fraction': '0.03831'
    }, {
        'site': '10',
        'defect_fraction': '0.00151501010'
    }, {
        'site': '11',
        'defect_fraction': '0.00479'
    }, {
        'site': '12',
        'defect_fraction': '0.00036'
    }, {
        'site': '13',
        'defect_fraction': '0.00219'
    }, {
        'site': '14',
        'defect_fraction': '0.0041'
    }, {
        'site': '15',
        'defect_fraction': '0.005250'
    }, {
        'site': '16',
        'defect_fraction': '0.00242'
    }, {
        'site': '17',
        'defect_fraction': '0.002630'
    }, {
        'site': '18',
        'defect_fraction': '0.00013'
    }, None, {
        'site': '1',
        'defect_fraction': '0.13052'
    }, {
        'site': '20',
        'defect_fraction': '0.0'
    }, {
        'site': '21',
        'defect_fraction': '0.00243'
    }, {
        'site': '22',
        'defect_fraction': '0.0026410'
    }, {
        'site': '23',
        'defect_fraction': '0.00249'
    }, {
        'site': '24',
        'defect_fraction': '0.003810'
    }, None, {
        'site': '26',
        'defect_fraction': '0.000550'
    }, None, {
        'site': '28',
        'defect_fraction': '0.001730'
    }, {
        'site': '29',
        'defect_fraction': '0.00029'
    }, {
        'site': '2',
        'defect_fraction': '0.02934'
    }, {
        'site': '30',
        'defect_fraction': '0.000660'
    }, {
        'site': '31',
        'defect_fraction': '0.00019'
    }, {
        'site': '32',
        'defect_fraction': '0.00035'
    }, {
        'site': '33',
        'defect_fraction': '0.0001000'
    }, {
        'site': '34',
        'defect_fraction': '0.0003'
    }, {
        'site': '35',
        'defect_fraction': '0.0'
    }, {
        'site': '36',
        'defect_fraction': '0.0'
    }, {
        'site': '37',
        'defect_fraction': '5e-05'
    }, {
        'site': '38',
        'defect_fraction': '0.00'
    }, {
        'site': '39',
        'defect_fraction': '6e-05'
    }, {
        'site': '3',
        'defect_fraction': '0.02597'
    }, {
        'site': '40',
        'defect_fraction': '0.0001'
    }, None, {
        'site': '42',
        'defect_fraction': '5e-05'
    }, None, {
        'site': '44',
        'defect_fraction': '0.0'
    }, None, {
        'site': '46',
        'defect_fraction': '0.0001810'
    }, None, {
        'site': '48',
        'defect_fraction': '0.01010020020'
    }, None, {
        'site': '4',
        'defect_fraction': '0.02745'
    }, {
        'site': '5',
        'defect_fraction': '0.02307'
    }, {
        'site': '6',
        'defect_fraction': '0.02731'
    }, {
        'site': '7',
        'defect_fraction': '0.0377'
    }, {
        'site': '8',
        'defect_fraction': '0.0272'
    }, {
        'site': '9',
        'defect_fraction': '0.02521'
    }, {
        'site': '10',
        'defect_fraction': '0.00385190'
    }, {
        'site': '11',
        'defect_fraction': '0.00549'
    }, {
        'site': '12',
        'defect_fraction': '0.00033'
    }, {
        'site': '13',
        'defect_fraction': '0.00034'
    }, {
        'site': '14',
        'defect_fraction': '0.0038510'
    }, None, {
        'site': '16',
        'defect_fraction': '0.00168'
    }, {
        'site': '17',
        'defect_fraction': '0.00407'
    }, {
        'site': '18',
        'defect_fraction': '0.00405'
    }, None, {
        'site': '1',
        'defect_fraction': '0.12495'
    }, None, {
        'site': '21',
        'defect_fraction': '0.00306'
    }, {
        'site': '22',
        'defect_fraction': '0.00372'
    }, {
        'site': '23',
        'defect_fraction': '0.004430'
    }, {
        'site': '24',
        'defect_fraction': '0.00217'
    }, {
        'site': '25',
        'defect_fraction': '0.00013'
    }, {
        'site': '26',
        'defect_fraction': '0.01010020020'
    }, {
        'site': '27',
        'defect_fraction': '0.0'
    }, {
        'site': '28',
        'defect_fraction': '0.00118'
    }, {
        'site': '29',
        'defect_fraction': '0.000370'
    }, {
        'site': '2',
        'defect_fraction': '0.03829'
    }, {
        'site': '30',
        'defect_fraction': '0.00217'
    }, {
        'site': '31',
        'defect_fraction': '0.00037'
    }, {
        'site': '32',
        'defect_fraction': '0.001150'
    }, {
        'site': '33',
        'defect_fraction': '0.00071'
    }, {
        'site': '4',
        'defect_fraction': '0.0003620'
    }, {
        'site': '35',
        'defect_fraction': '0.0'
    }, {
        'site': '36',
        'defect_fraction': '0.0'
    }, {
        'site': '37',
        'defect_fraction': '5e-050'
    }, {
        'site': '38',
        'defect_fraction': '0.00103'
    }, {
        'site': '39',
        'defect_fraction': '0.00035'
    }, {
        'site': '3',
        'defect_fraction': '0.022190'
    }, {
        'site': '40',
        'defect_fraction': '0.0002527'
    }, {
        'site': '41',
        'defect_fraction': '6e-05'
    }, {
        'site': '42',
        'defect_fraction': '0.00030'
    }, {
        'site': '43',
        'defect_fraction': '0.020'
    }, {
        'site': '44',
        'defect_fraction': '6e-05'
    }, {
        'site': '45',
        'defect_fraction': '0.000'
    }, {
        'site': '46',
        'defect_fraction': '0.01904'
    }, {
        'site': '47',
        'defect_fraction': '0.0'
    }, {
        'site': '48',
        'defect_fraction': '6e-0510'
    }, {
        'site': '9',
        'defect_fraction': '0.0190'
    }, {
        'site': '4',
        'defect_fraction': '0.02662'
    }, {
        'site': '5',
        'defect_fraction': '0.02391'
    }, {
        'site': '6',
        'defect_fraction': '0.018720'
    }, {
        'site': '7',
        'defect_fraction': '0.0204'
    }, {
        'site': '8',
        'defect_fraction': '0.03008'
    }, {
        'site': '9',
        'defect_fraction': '0.03086'
    }, {
        'site': '10',
        'defect_fraction': '0.000691010020020'
    }, None, {
        'site': '12',
        'defect_fraction': '0.00174'
    }, {
        'site': '13',
        'defect_fraction': '5e-05'
    }, {
        'site': '14',
        'defect_fraction': '0.00224'
    }, {
        'site': '15',
        'defect_fraction': '0.00214'
    }, {
        'site': '16',
        'defect_fraction': '0.000365'
    }, {
        'site': '17',
        'defect_fraction': '0.00017'
    }, None, {
        'site': '19',
        'defect_fraction': '0.00022'
    }, {
        'site': '1',
        'defect_fraction': '0.08438'
    }, {
        'site': '20',
        'defect_fraction': '0.00036'
    }, {
        'site': '21',
        'defect_fraction': '0.001850'
    }, {
        'site': '22',
        'defect_fraction': '0.00076'
    }, {
        'site': '23',
        'defect_fraction': '0.001380'
    }, {
        'site': '24',
        'defect_fraction': '0.00067'
    }, {
        'site': '25',
        'defect_fraction': '0.00135'
    }, None, {
        'site': '27',
        'defect_fraction': '5e-05'
    }, {
        'site': '28',
        'defect_fraction': '0.00014'
    }, {
        'site': '29',
        'defect_fraction': '0.01905'
    }, {
        'site': '2',
        'defect_fraction': '0.01416'
    }, {
        'site': '30',
        'defect_fraction': '0.20911'
    }, {
        'site': '31',
        'defect_fraction': '0.0'
    }, {
        'site': '32',
        'defect_fraction': '0.074861'
    }, {
        'site': '33',
        'defect_fraction': '0.24568'
    }, {
        'site': '34',
        'defect_fraction': '0.00456'
    }, {
        'site': '35',
        'defect_fraction': '0.080450'
    }, {
        'site': '36',
        'defect_fraction': '0.0'
    }, {
        'site': '37',
        'defect_fraction': '0.02401'
    }, None, {
        'site': '39',
        'defect_fraction': '0.06734'
    }, {
        'site': '3',
        'defect_fraction': '0.01804'
    }, {
        'site': '40',
        'defect_fraction': '0.20601'
    }, {
        'site': '42',
        'defect_fraction': '0.00593'
    }, {
        'site': '42',
        'defect_fraction': '5e-05'
    }, {
        'site': '43',
        'defect_fraction': '5e-050'
    }, None, {
        'site': '45',
        'defect_fraction': '0.00124'
    }, {
        'site': '46',
        'defect_fraction': '0.00'
    }, None, None, None, {
        'site': '4',
        'defect_fraction': '0.02112'
    }, {
        'site': '5',
        'defect_fraction': '0.00225'
    }, {
        'site': '6',
        'defect_fraction': '0.01649'
    }, {
        'site': '7',
        'defect_fraction': '0.02151'
    }, {
        'site': '8',
        'defect_fraction': '0.01739'
    }, {
        'site': '9',
        'defect_fraction': '0.017940'
    }, {
        'site': '10',
        'defect_fraction': '7e-05'
    }, None, {
        'site': '12',
        'defect_fraction': '0.0088800'
    }, {
        'site': '13',
        'defect_fraction': '0.000120'
    }, {
        'site': '14',
        'defect_fraction': '0.00571'
    }, {
        'site': '15',
        'defect_fraction': '7e-05'
    }, {
        'site': '16',
        'defect_fraction': '0.000240'
    }, {
        'site': '17',
        'defect_fraction': '0.00016'
    }, {
        'site': '18',
        'defect_fraction': '0.00618'
    }, None, {
        'site': '1',
        'defect_fraction': '0.07701'
    }, {
        'site': '20',
        'defect_fraction': '0.00356'
    }, {
        'site': '21',
        'defect_fraction': '0.00292'
    }, {
        'site': '22',
        'defect_fraction': '0.0'
    }, {
        'site': '23',
        'defect_fraction': '0.00292'
    }, {
        'site': '24',
        'defect_fraction': '0.00043'
    }, {
        'site': '25',
        'defect_fraction': '0.00086'
    }, None, {
        'site': '27',
        'defect_fraction': '0.00015'
    }, {
        'site': '28',
        'defect_fraction': '0.00283'
    }, {
        'site': '29',
        'defect_fraction': '0.002260'
    }, {
        'site': '2',
        'defect_fraction': '0.01558'
    }, {
        'site': '30',
        'defect_fraction': '0.00311'
    }, {
        'site': '31',
        'defect_fraction': '0.00095'
    }, {
        'site': '32',
        'defect_fraction': '0.00127'
    }, {
        'site': '33',
        'defect_fraction': '0.001520'
    }, {
        'site': '34',
        'defect_fraction': '0.000470'
    }, {
        'site': '35',
        'defect_fraction': '0.000420'
    }, {
        'site': '36',
        'defect_fraction': '0.0004600'
    }, {
        'site': '37',
        'defect_fraction': '0.00018'
    }, {
        'site': '38',
        'defect_fraction': '0.00095'
    }, {
        'site': '39',
        'defect_fraction': '0.0005'
    }, {
        'site': '3',
        'defect_fraction': '0.01677'
    }, {
        'site': '40',
        'defect_fraction': '0.0'
    }, {
        'site': '41',
        'defect_fraction': '0.010'
    }, None, None, {
        'site': '44',
        'defect_fraction': '6e-05'
    }, {
        'site': '45',
        'defect_fraction': '62'
    }, None, None, {
        'site': '48',
        'defect_fraction': '0.0'
    }, {
        'site': '40',
        'defect_fraction': '0.020'
    }, {
        'site': '4',
        'defect_fraction': '0.02924'
    }, {
        'site': '5',
        'defect_fraction': '0.02174'
    }, {
        'site': '6',
        'defect_fraction': '0.01306'
    }, {
        'site': '7',
        'defect_fraction': '0.0177'
    }, {
        'site': '8',
        'defect_fraction': '0.02332'
    }, {
        'site': '9',
        'defect_fraction': '0.037662'
    }, {
        'site': '10',
        'defect_fraction': '0.0010500'
    }, {
        'site': '11',
        'defect_fraction': '0.00408'
    }, {
        'site': '12',
        'defect_fraction': '0.00401'
    }, None, {
        'site': '14',
        'defect_fraction': '0.00097'
    }, {
        'site': '15',
        'defect_fraction': '0.0016'
    }, {
        'site': '16',
        'defect_fraction': '0.00273'
    }, {
        'site': '17',
        'defect_fraction': '0.00293'
    }, None, {
        'site': '19',
        'defect_fraction': '0.00094'
    }, {
        'site': '1',
        'defect_fraction': '0.0933'
    }, None, {
        'site': '21',
        'defect_fraction': '0.0014'
    }, {
        'site': '22',
        'defect_fraction': '0.00179'
    }, None, None, None, None, {
        'site': '27',
        'defect_fraction': '0.00011'
    }, {
        'site': '28',
        'defect_fraction': '0.002330'
    }, {
        'site': '29',
        'defect_fraction': '0.00163'
    }, {
        'site': '2',
        'defect_fraction': '0.016780'
    }, {
        'site': '30',
        'defect_fraction': '0.00204'
    }, {
        'site': '31',
        'defect_fraction': '0.0003'
    }, {
        'site': '32',
        'defect_fraction': '0.00072'
    }, {
        'site': '33',
        'defect_fraction': '0.000690'
    }, {
        'site': '34',
        'defect_fraction': '0.000260'
    }, {
        'site': '35',
        'defect_fraction': '0.00026'
    }, {
        'site': '36',
        'defect_fraction': '0.00019'
    }, {
        'site': '37',
        'defect_fraction': '82'
    }, {
        'site': '38',
        'defect_fraction': '0.00131'
    }, {
        'site': '39',
        'defect_fraction': '0.0'
    }, {
        'site': '3',
        'defect_fraction': '0.02443'
    }, {
        'site': '40',
        'defect_fraction': '5e-05'
    }, {
        'site': '41',
        'defect_fraction': '5e-05'
    }, {
        'site': '42',
        'defect_fraction': '0.000100'
    }, {
        'site': '43',
        'defect_fraction': '0.0001'
    }, None, {
        'site': '45',
        'defect_fraction': '0.00047'
    }, {
        'site': '46',
        'defect_fraction': '5e-0510'
    }, None, None, {
        'site': '49',
        'defect_fraction': '0.0'
    }, {
        'site': '4',
        'defect_fraction': '0.01723'
    }, None, {
        'site': '6',
        'defect_fraction': '0.01569'
    }, {
        'site': '7',
        'defect_fraction': '0.02011'
    }, {
        'site': '8',
        'defect_fraction': '0.02839'
    }, {
        'site': '9',
        'defect_fraction': '0.0143'
    }, None, None, {
        'site': '12',
        'defect_fraction': '0.0023'
    }, {
        'site': '13',
        'defect_fraction': '0.00055'
    }, {
        'site': '14',
        'defect_fraction': '0.00119'
    }, {
        'site': '15',
        'defect_fraction': '0.00616'
    }, None, {
        'site': '17',
        'defect_fraction': '0.00196'
    }, {
        'site': '18',
        'defect_fraction': '0.0024300'
    }, None, {
        'site': '1',
        'defect_fraction': '0.09019'
    }, {
        'site': '20',
        'defect_fraction': '0.0'
    }, None, {
        'site': '22',
        'defect_fraction': '0.00103'
    }, {
        'site': '23',
        'defect_fraction': '0.00'
    }, {
        'site': '24',
        'defect_fraction': '0.002'
    }, None, {
        'site': '26',
        'defect_fraction': '5e-05'
    }, {
        'site': '27',
        'defect_fraction': '0.00011'
    }, {
        'site': '28',
        'defect_fraction': '0.00172'
    }, {
        'site': '29',
        'defect_fraction': '0.00'
    }, {
        'site': '2',
        'defect_fraction': '0.01879'
    }, {
        'site': '30',
        'defect_fraction': '0.0007'
    }, {
        'site': '31',
        'defect_fraction': '0.000530'
    }, {
        'site': '32',
        'defect_fraction': '0.00049'
    }, {
        'site': '33',
        'defect_fraction': '0.0003'
    }, {
        'site': '34',
        'defect_fraction': '0.000210'
    }, {
        'site': '35',
        'defect_fraction': '7e-050'
    }, {
        'site': '36',
        'defect_fraction': '0.00022'
    }, {
        'site': '37',
        'defect_fraction': '0.00012'
    }, {
        'site': '38',
        'defect_fraction': '9e-05'
    }, {
        'site': '39',
        'defect_fraction': '0.00011'
    }, {
        'site': '3',
        'defect_fraction': '0.01713'
    }, {
        'site': '40',
        'defect_fraction': '0.0001'
    }, None, {
        'site': '42',
        'defect_fraction': '0.00027'
    }, {
        'site': '43',
        'defect_fraction': '0.00066'
    }, {
        'site': '44',
        'defect_fraction': '0.000'
    }, {
        'site': '45',
        'defect_fraction': '9e-05'
    }, {
        'site': '46',
        'defect_fraction': '0.000120'
    }, None, None, {
        'site': '49',
        'defect_fraction': '5e-05'
    }, {
        'site': '4',
        'defect_fraction': '0.03003'
    }, {
        'site': '5',
        'defect_fraction': '0.01433'
    }, {
        'site': '6',
        'defect_fraction': '0.01621'
    }, {
        'site': '7',
        'defect_fraction': '0.0151'
    }, {
        'site': '8',
        'defect_fraction': '0.03413'
    }, {
        'site': '9',
        'defect_fraction': '0.02138'
    }, None, None, {
        'site': '12',
        'defect_fraction': '0.00303'
    }, {
        'site': '13',
        'defect_fraction': '0.001240'
    }, {
        'site': '14',
        'defect_fraction': '0.001220'
    }, {
        'site': '15',
        'defect_fraction': '0.00704'
    }, {
        'site': '16',
        'defect_fraction': '0.00064'
    }, None, {
        'site': '18',
        'defect_fraction': '0.00453'
    }, None, {
        'site': '1',
        'defect_fraction': '0.07717'
    }, None, {
        'site': '21',
        'defect_fraction': '0.00574'
    }, {
        'site': '22',
        'defect_fraction': '0.00545'
    }, None, None, None, {
        'site': '26',
        'defect_fraction': '0.0'
    }, {
        'site': '27',
        'defect_fraction': '0.0'
    }, {
        'site': '28',
        'defect_fraction': '0.00162'
    }, None, {
        'site': '2',
        'defect_fraction': '0.03816'
    }, {
        'site': '30',
        'defect_fraction': '0.00166'
    }, None, {
        'site': '32',
        'defect_fraction': '0.000750'
    }, {
        'site': '33',
        'defect_fraction': '0.000840'
    }, {
        'site': '34',
        'defect_fraction': '0.00011'
    }, {
        'site': '35',
        'defect_fraction': '0.00038'
    }, {
        'site': '36',
        'defect_fraction': '7e-050'
    }, {
        'site': '37',
        'defect_fraction': '0.00012'
    }, {
        'site': '38',
        'defect_fraction': '0.0'
    }, {
        'site': '39',
        'defect_fraction': '0.00066'
    }, {
        'site': '3',
        'defect_fraction': '0.03733'
    }, {
        'site': '40',
        'defect_fraction': '0.00029'
    }, {
        'site': '41',
        'defect_fraction': '0.00084'
    }, {
        'site': '42',
        'defect_fraction': '0.00023'
    }, None, {
        'site': '44',
        'defect_fraction': '0.00'
    }, {
        'site': '45',
        'defect_fraction': '5e-05'
    }, {
        'site': '46',
        'defect_fraction': '0.0'
    }, {
        'site': '47',
        'defect_fraction': '0.0'
    }, None, None, {
        'site': '4',
        'defect_fraction': '0.035040'
    }, {
        'site': '5',
        'defect_fraction': '0.03126'
    }, {
        'site': '6',
        'defect_fraction': '0.03325'
    }, {
        'site': '7',
        'defect_fraction': '0.04945'
    }, {
        'site': '8',
        'defect_fraction': '0.04756'
    }, {
        'site': '9',
        'defect_fraction': '0.06265'
    }, {
        'site': '10',
        'defect_fraction': '0.00731'
    }, {
        'site': '11',
        'defect_fraction': '0.00314'
    }, None, {
        'site': '13',
        'defect_fraction': '0.00013'
    }, {
        'site': '14',
        'defect_fraction': '0.01008'
    }, None, {
        'site': '16',
        'defect_fraction': '0.00430'
    }, {
        'site': '17',
        'defect_fraction': '0.01036'
    }, {
        'site': '18',
        'defect_fraction': '0.00084'
    }, {
        'site': '19',
        'defect_fraction': '0.00179'
    }, {
        'site': '1',
        'defect_fraction': '0.13869'
    }, {
        'site': '20',
        'defect_fraction': '0.00443'
    }, {
        'site': '21',
        'defect_fraction': '0.00584'
    }, None, {
        'site': '23',
        'defect_fraction': '0.00928'
    }, {
        'site': '24',
        'defect_fraction': '0.01323'
    }, {
        'site': '25',
        'defect_fraction': '0.0025'
    }, {
        'site': '26',
        'defect_fraction': '0.0'
    }, None, {
        'site': '28',
        'defect_fraction': '0.00250'
    }, {
        'site': '29',
        'defect_fraction': '0.000450'
    }, {
        'site': '2',
        'defect_fraction': '0.02838'
    }, {
        'site': '30',
        'defect_fraction': '0.001820'
    }, {
        'site': '31',
        'defect_fraction': '0.00047'
    }, {
        'site': '32',
        'defect_fraction': '0.001980'
    }, {
        'site': '33',
        'defect_fraction': '0.001240'
    }, {
        'site': '34',
        'defect_fraction': '0.00061'
    }, {
        'site': '35',
        'defect_fraction': '0.000960'
    }, {
        'site': '36',
        'defect_fraction': '0.00041'
    }, None, {
        'site': '38',
        'defect_fraction': '0.00016'
    }, {
        'site': '39',
        'defect_fraction': '0.00024'
    }, {
        'site': '3',
        'defect_fraction': '0.03215'
    }, None, {
        'site': '41',
        'defect_fraction': '0.00'
    }, None, None, {
        'site': '44',
        'defect_fraction': '0.00182'
    }, {
        'site': '45',
        'defect_fraction': '0.00011'
    }, {
        'site': '46',
        'defect_fraction': '5e-05'
    }, {
        'site': '47',
        'defect_fraction': '0.0'
    }, None, None, {
        'site': '4',
        'defect_fraction': '0.03679'
    }, {
        'site': '5',
        'defect_fraction': '0.05891'
    }, {
        'site': '6',
        'defect_fraction': '0.03872'
    }, {
        'site': '7',
        'defect_fraction': '0.050620'
    }, {
        'site': '8',
        'defect_fraction': '0.052710'
    }, {
        'site': '9',
        'defect_fraction': '0.04195'
    }, None, None, {
        'site': '12',
        'defect_fraction': '0.00518'
    }, {
        'site': '13',
        'defect_fraction': '0.00041'
    }, {
        'site': '14',
        'defect_fraction': '0.00207000'
    }, {
        'site': '15',
        'defect_fraction': '0.00993'
    }, {
        'site': '16',
        'defect_fraction': '0.00066'
    }, {
        'site': '17',
        'defect_fraction': '0.00384'
    }, None, None, {
        'site': '1',
        'defect_fraction': '0.08751'
    }, None, {
        'site': '21',
        'defect_fraction': '0.00526'
    }, {
        'site': '22',
        'defect_fraction': '0.00123'
    }, None, {
        'site': '24',
        'defect_fraction': '0.00458'
    }, None, None, {
        'site': '27',
        'defect_fraction': '0.00072'
    }, {
        'site': '28',
        'defect_fraction': '0.001580'
    }, {
        'site': '29',
        'defect_fraction': '0.0003410'
    }, {
        'site': '2',
        'defect_fraction': '0.01956'
    }, {
        'site': '30',
        'defect_fraction': '0.002090'
    }, {
        'site': '31',
        'defect_fraction': '0.00203'
    }, {
        'site': '32',
        'defect_fraction': '0.00085'
    }, {
        'site': '33',
        'defect_fraction': '0.00063'
    }, {
        'site': '34',
        'defect_fraction': '6e-05'
    }, {
        'site': '35',
        'defect_fraction': '0.001250'
    }, {
        'site': '36',
        'defect_fraction': '0.36554'
    }, None, {
        'site': '38',
        'defect_fraction': '0.0'
    }, {
        'site': '39',
        'defect_fraction': '7e-05'
    }, {
        'site': '3',
        'defect_fraction': '0.02763'
    }, {
        'site': '40',
        'defect_fraction': '0.00016'
    }, {
        'site': '41',
        'defect_fraction': '0.00375'
    }, {
        'site': '42',
        'defect_fraction': '9e-05'
    }, None, {
        'site': '44',
        'defect_fraction': '0.0'
    }, {
        'site': '45',
        'defect_fraction': '0.00'
    }, None, None, {
        'site': '48',
        'defect_fraction': '0.2604'
    }, {
        'site': '49',
        'defect_fraction': '0.19184'
    }, {
        'site': '4',
        'defect_fraction': '0.03064'
    }, {
        'site': '5',
        'defect_fraction': '0.0309'
    }, {
        'site': '6',
        'defect_fraction': '0.03093'
    }, {
        'site': '7',
        'defect_fraction': '0.03343'
    }, {
        'site': '8',
        'defect_fraction': '0.026392'
    }, {
        'site': '9',
        'defect_fraction': '0.03186'
    }]

    wafer_map_excel(site_defect_fraction_data=site_defect_fraction_data)


main()

# image_output.read()  # Do as you wish with it!

# image = Image.frombytes('RGB',(300,300),b64decode(img_src_padding))
# image.save("./test/foo.png")

# with open("./test/imageToSavee.png", "wb") as fh:
# fh.write(base64.b64decode(img_src_padding))

# with open("./test/test.png","wb") as f:
# img_data = base64.b64decode(img_src_padding)
# bytes = base64.decodebytes(bytes(img_src_padding, "utf-8"))
# f.write(img_data)

# f = open('test.jpg', 'wb')

# f.write(bytes)

# f.write(bytes(img_src_padding, "utf-8"))
# f.close()

# def main(url, out_folder="./merrychristmas/"):
#     """Downloads all the images at 'url' to /test/"""
#     soup = document
#     parsed = list(urlparse(url))

#     for image in soup.findAll("img"):
#         # print("Image: %(src)s" % image)
#         filename = image["src"].split("/")[-1]
#         parsed[2] = image["src"]
#         outpath = os.path.join(out_folder, filename[0:5] )

#         urlretrieve(image["src"], outpath)

# if __name__ == "__main__":
#     url = sys.argv[-1]
#     out_folder = "/test/"

#     main(out_folder)

# Defining binary path to the pytesseract library
# pytesseract.pytesseract.tesseract_cmd = "./tesseract-ocr-w32-setup-v5.0.0.20211201.exe"

# for index, img in enumerate(document.find_all('img')):
#     if index == 1:
#         img_src =img['src']
#         img_src_padding  = f"{img_src}{'=' * (len(img_src) % 4)}"

#         img = Image.frombytes(mode="RGB",size=(300,300), data=base64.decodebytes(bytes(img_src_padding, "utf-8")))
#         img.show()

# img_data = base64.b64decode(img_src_padding)

# image = Image.open(io.BytesIO(img_data))
# image.show()

# image = Image.frombytes("L", (800,200), img_data)

# # Create in-memory PNG
# buffer = io.BytesIO()
# image.save(buffer, format="PNG")
# PNG = buffer.getvalue()

# image = cv2.imshow("Image", np.array(image))
# cv2.waitKey(0)

# text = pytesseract.image_to_string(PNG)
# print(text)

# print(img)
