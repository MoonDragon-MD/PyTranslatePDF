# PyTranslatePDF
# V 1.0 rev56
# By MoonDragon (https://github.com/MoonDragon-MD/PyTranslatePDF)
# Dipendence
# sudo apt install fonts-dejavu (with Linux)
# pip install PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf
# or out of 3.10 Portable
# python -m pip install PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf
# use with
# python PyTranslatePDF.py input.pdf output.pdf en it [font_customized] [translate_locally_path]

import fitz  # PyMuPDF
import requests
import sys
import logging
import os
import platform
from pathlib import Path
import subprocess
from googletrans import Translator  # Synchronic version of googletrans
import numpy as np
from PIL import Image
import io
import pdfrw
import pikepdf
import json
import tempfile
import shutil
import re

TRANSLATE_LOCALLY_MODELS = None
TRANSLATE_LOCALLY_AVAILABLE = False
TRANSLATE_LOCALLY_PATH = None

# Configurazione del logging
LOGGING = False # Set to True or False to turn on or off logging
if LOGGING:
    logging.basicConfig(filename='PyTranslatePDF_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.CRITICAL)

LIBRETRANSLATE_URL = "http://localhost:5000/translate"
NLLB_SERVE_URL = "http://127.0.0.1:6060/translate"

def check_translate_locally_availability(custom_path=None):
    global TRANSLATE_LOCALLY_AVAILABLE, TRANSLATE_LOCALLY_MODELS, TRANSLATE_LOCALLY_PATH
    
    if TRANSLATE_LOCALLY_AVAILABLE is not None and TRANSLATE_LOCALLY_MODELS is not None:
        return TRANSLATE_LOCALLY_AVAILABLE

    # On Ubuntu/Linux, try using "translateLocally" on the PATH
    if platform.system() != "Windows":
        try:
            result_list = subprocess.run(
                ["translateLocally", "-l"],
                capture_output=True,
                text=True,
                check=False
            )
            if result_list.returncode == 0:
                TRANSLATE_LOCALLY_PATH = "translateLocally"  # Use command from PATH
            else:
                logging.warning(f"translateLocally not found in the PATH: {result_list.stderr}. Disabilitato.")
                TRANSLATE_LOCALLY_AVAILABLE = False
                return False
        except FileNotFoundError:
            logging.warning("translateLocally not found in the PATH on Ubuntu/Linux. Disabled.")
            TRANSLATE_LOCALLY_AVAILABLE = False
            return False
    else:  # Su Windows, usa il custom_path se fornito
        if not custom_path:
            logging.warning("On Windows, you need to specify the path translateLocally. Disabled.")
            TRANSLATE_LOCALLY_AVAILABLE = False
            return False

        TRANSLATE_LOCALLY_PATH = custom_path
        
        if not os.path.isfile(TRANSLATE_LOCALLY_PATH):
            logging.warning(f"File translateLocally not found in: {TRANSLATE_LOCALLY_PATH} on Windows. Disabled.")
            TRANSLATE_LOCALLY_AVAILABLE = False
            return False

        try:
            result_list = subprocess.run(
                [TRANSLATE_LOCALLY_PATH, "-l"],
                capture_output=True,
                text=True,
                check=False,
                shell=True  # It may be necessary on Windows
            )
            if result_list.returncode != 0:
                logging.warning(f"translateLocally not working on Windows: {result_list.stderr}. Disabilitato.")
                TRANSLATE_LOCALLY_AVAILABLE = False
                return False
        except Exception as e:
            logging.warning(f"Errore nel verificare translateLocally su Windows: {e}. Disabled.")
            TRANSLATE_LOCALLY_AVAILABLE = False
            return False

    # Try to parse the models
    try:
        language_models = {}
        output_lines = result_list.stdout.strip().splitlines()
        for line in output_lines:
            if "To invoke do -m" in line and any(model in line.lower() for model in ["tiny", "base", "transformer-tiny11", "full"]):
                start_idx = line.find("do -m") + 6
                if start_idx != -1 and start_idx < len(line):
                    translation_spec = line[start_idx:].strip().split("-")
                    if len(translation_spec) >= 3:
                        source_lang = translation_spec[0].lower()
                        target_lang = translation_spec[1].lower()
                        model = translation_spec[2].lower()
                        language_map = {
                            "afrikaans": "af", "arabic": "ar", "bulgarian": "bg", "catalan": "ca", "czech": "cs",
                            "german": "de", "greek": "el", "english": "en", "spanish": "es", "estonian": "et",
                            "basque": "eu", "french": "fr", "galician": "gl", "serbo-croatian": "hbs", "hebrew": "he",
                            "hindi": "hi", "icelandic": "is", "italian": "it", "japanese": "ja", "korean": "ko",
                            "macedonian": "mk", "malay": "ml", "maltese": "mt", "norwegian": "no", "polish": "pl",
                            "sinhala": "si", "slovak": "sk", "slovene": "sl", "albanian": "sq", "swahili": "sw",
                            "thai": "th", "turkish": "tr", "ukrainian": "uk", "vietnamese": "vi"
                        }
                        normalized_source = next((code for full, code in language_map.items() if full in source_lang), source_lang)
                        normalized_target = next((code for full, code in language_map.items() if full in target_lang), target_lang)
                        if normalized_source not in language_models:
                            language_models[normalized_source] = []
                        language_models[normalized_source].append((normalized_target, model))

        TRANSLATE_LOCALLY_MODELS = language_models
        TRANSLATE_LOCALLY_AVAILABLE = True
        logging.debug(f"Models and languages available initialized: {TRANSLATE_LOCALLY_MODELS}")
        return True
    except Exception as e:
        logging.warning(f"Error in parsing models translateLocally: {e}. Disabilitato.")
        TRANSLATE_LOCALLY_AVAILABLE = False
        return False

def is_translate_locally_available():
    return TRANSLATE_LOCALLY_AVAILABLE

def translate_with_libretranslate(text, source, target):
    try:
        response = requests.post(
            LIBRETRANSLATE_URL,
            json={"q": text, "source": source, "target": target, "format": "text"}
        )
        response.raise_for_status()
        translated = response.json().get("translatedText", text)
        logging.debug(f"Translated by LibreTranslate: {translated}")
        return translated
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in translation LibreTranslate: {e}")
        raise

def translate_locally(text, source, target):
    if not TRANSLATE_LOCALLY_AVAILABLE:
        raise ValueError("translateLocally is not available on this system.")

    language_models = TRANSLATE_LOCALLY_MODELS

    if source.lower() not in language_models:
        raise ValueError(f"No model available for the language of origin: {source}")

    direct_translation = next((model for target_lang, model in language_models[source.lower()] if target_lang == target.lower()), None)
    if direct_translation:
        max_chunk_size = 500
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_chunks = []

        for chunk in chunks:
            if not chunk.strip() or all(ord(c) in {0x200B, 0xFEFF, 0x00A0, 32} for c in chunk):
                translated_chunks.append("")
                logging.debug(f"Empty or invisible Chunk, ignored: {repr(chunk)}")
                continue

            command = [TRANSLATE_LOCALLY_PATH, "-m", f"{source.lower()}-{target.lower()}-{direct_translation}"]
            logging.info(f"Esecuzione di translateLocally: {' '.join(command)} con chunk: {repr(chunk)[:100]}...")
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=chunk)
            
            if process.returncode == 0:
                translated_chunk = stdout.strip()
                translated_chunk = "\n".join(line for line in translated_chunk.splitlines() if not line.startswith("QVariant"))
                if translated_chunk:
                    translated_chunks.append(translated_chunk)
                    logging.debug(f"Chunk translated: {translated_chunk[:100]}...")
                else:
                    logging.warning(f"Output empty for trust chunk: {repr(chunk)} (hex: {chunk.encode('utf-8').hex()}) stderr: {stderr}")
                    translated_chunks.append("")
            else:
                logging.error(f"Direct translation error: {stderr}")
                raise ValueError(f"Direct translation error for chunk: {repr(chunk)} - stderr: {stderr}")

        translated_text = " ".join(translated_chunks)
        logging.debug(f"Translated by translateLocally: {translated_text[:100]}...")
        return translated_text

    intermediate_lang = "en"
    if intermediate_lang not in language_models and intermediate_lang not in [t for s, t_m in language_models.items() for t, _ in t_m]:
        raise ValueError(f"Intermediate language {intermediate_lang} not supported")

    if source.lower() in language_models:
        en_model = next((model for t, model in language_models[source.lower()] if t == intermediate_lang), None)
        if en_model:
            command = [TRANSLATE_LOCALLY_PATH, "-m", f"{source.lower()}-{intermediate_lang}-{en_model}"]
            logging.info(f"Running translateLocally (source -> en): {' '.join(command)}")
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0:
                intermediate_text = stdout.strip()
                intermediate_text = "\n".join(line for line in intermediate_text.splitlines() if not line.startswith("QVariant"))
                if not intermediate_text:
                    raise ValueError("Output of translateLocally not valid for the first translation.")
            else:
                raise ValueError(f"Error in the first translation: {stderr}")
        else:
            raise ValueError(f"No model found for {source.lower()} -> {intermediate_lang}")
    else:
        raise ValueError(f"Origin language {source.lower()} not supported")

    if intermediate_lang in language_models:
        target_model = next((model for t, model in language_models[intermediate_lang] if t == target.lower()), None)
        if target_model:
            command = [TRANSLATE_LOCALLY_PATH, "-m", f"{intermediate_lang}-{target.lower()}-{target_model}"]
            logging.info(f"Running translateLocally (en -> target): {' '.join(command)}")
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=intermediate_text)
            
            if process.returncode == 0:
                translated_text = stdout.strip()
                translated_text = "\n".join(line for line in translated_text.splitlines() if not line.startswith("QVariant"))
                if translated_text:
                    logging.debug(f"Translated by translateLocally: {translated_text}")
                    return translated_text
                else:
                    raise ValueError("Output of translateLocally non valid for the second translation.")
            else:
                raise ValueError(f"Error in second translation: {stderr}")
        else:
            raise ValueError(f"No model found for {intermediate_lang} -> {target.lower()}")
    else:
        raise ValueError(f"Intermediate language {intermediate_lang} not found in models")

def translate_with_nllb(text, source_lang, target_lang):
    """Translate using NLLB-serve with language mapping"""
    def format_language(lang):
        """Language format for NLLB-serve"""
        language_mapping = {
            'af': 'afr_Latn', 'ak': 'aka_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab', 'as': 'asm_Beng',
            'ay': 'ayr_Latn', 'az': 'azj_Latn', 'be': 'bel_Cyrl', 'bg': 'bul_Cyrl', 'bn': 'ben_Beng',
            'bs': 'bos_Latn', 'ca': 'cat_Latn', 'cs': 'ces_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn',
            'de': 'deu_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn', 'eo': 'epo_Latn', 'es': 'spa_Latn',
            'et': 'est_Latn', 'eu': 'eus_Latn', 'fa': 'fao_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn',
            'ga': 'gla_Latn', 'gl': 'gle_Latn', 'gu': 'guj_Gujr', 'ha': 'hau_Latn', 'hi': 'hin_Deva',
            'hr': 'hrv_Latn', 'hu': 'hun_Latn', 'hy': 'hye_Armn', 'id': 'ind_Latn', 'is': 'isl_Latn',
            'it': 'ita_Latn', 'ja': 'jav_Latn', 'jw': 'jv_Latn', 'ka': 'kat_Geor', 'kk': 'kaz_Cyrl',
            'km': 'khm_Khmr', 'kn': 'kan_Knda', 'ko': 'kor_Hang', 'ku': 'kuw_Latn', 'ky': 'kir_Cyrl',
            'la': 'lat_Latn', 'lb': 'ltz_Latn', 'lg': 'lug_Latn', 'ln': 'lin_Latn', 'lo': 'lao_Laoo',
            'lt': 'lit_Latn', 'lv': 'lvs_Latn', 'mg': 'mlg_Latn', 'mi': 'mri_Latn', 'mk': 'mkd_Cyrl',
            'ml': 'mlt_Latn', 'mn': 'mnk_Cyrl', 'mr': 'mar_Deva', 'ms': 'msa_Latn', 'mt': 'mlt_Latn',
            'my': 'mya_Mymr', 'ne': 'nep_Deva', 'nl': 'nld_Latn', 'no': 'nob_Latn', 'nn': 'nno_Latn',
            'ny': 'nya_Latn', 'pa': 'pan_Guru', 'pl': 'pol_Latn', 'pt': 'por_Latn', 'qu': 'quy_Latn',
            'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'rw': 'kin_Latn', 'sa': 'san_Deva', 'sd': 'snd_Arab',
            'si': 'sin_Sinh', 'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sm': 'smo_Latn', 'sn': 'sna_Latn',
            'so': 'som_Latn', 'sq': 'sqi_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn', 'sw': 'swh_Latn',
            'ta': 'tam_Taml', 'te': 'tel_Telu', 'tg': 'tgk_Cyrl', 'th': 'tha_Thai', 'ti': 'tir_Ethi',
            'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vi': 'vie_Latn',
            'xh': 'xho_Latn', 'yi': 'yid_Hebr', 'yo': 'yor_Latn', 'zu': 'zul_Latn',
            'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant'
        }
        return language_mapping.get(lang.lower(), f"{lang}_Latn")

    url = NLLB_SERVE_URL
    params = {
        'source': [text],
        'src_lang': format_language(source_lang),
        'tgt_lang': format_language(target_lang),
    }
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(params))
        response.raise_for_status()
        data = response.json()
        translated = data.get('translation', '')
        # Make sure the result is a string, not a list
        if isinstance(translated, list):
            translated = translated[0] if translated else ''
        logging.debug(f"Tradotto da NLLB-serve: {translated}")
        return translated
    except Exception as e:
        logging.error(f"Errore con NLLB-serve: {str(e)}")
        raise

def translate_with_google(text, source_lang, target_lang):
    translator = Translator()
    try:
        translated = translator.translate(text, src=source_lang, dest=target_lang).text
        logging.debug(f"Translated by Google: {translated}")
        return translated
    except Exception as e:
        logging.error(f"Error with googletrans: {str(e)}")
        return text

def translate_text(text, source, target, engine):
    if engine == "T":
        return translate_with_libretranslate(text, source, target)
    elif engine == "L":
        return translate_locally(text, source, target)
    elif engine == "N":
        return translate_with_nllb(text, source, target)
    elif engine == "G":
        return translate_with_google(text, source, target)
    else:
        raise ValueError("Invalid translation engine")

def extract_font_info(page):
    font_info = []
    try:
        text_dict = page.get_text("dict")
    except AttributeError:
        logging.error("The get_text method is not available. Make sure to use a recent version of PyMuPDF (ex:1.19.3).")
        return font_info
    
    for block in text_dict["blocks"]:
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_info.append({
                        'font': span['font'],
                        'size': span['size'],
                        'text': span['text'],
                        'color': span['color'],
                        'bbox': span['bbox'], # Box coordinates (x0, y0, x1, y1)
                        'char_count': len(span['text'])
                    })
                    logging.debug(f"Font: {span['font']} - Size: {span['size']} - Colore: {span['color']} - Posizione: {span['bbox']}")
    return font_info

def calculate_font_size(original_text, translated_text, original_font_size):
    original_length = len(original_text)
    translated_length = len(translated_text)
    if translated_length <= original_length:
        return original_font_size
    scaling_factor = original_length / translated_length
    new_font_size = max(8, original_font_size * scaling_factor)
    logging.debug(f"Font size for translation: {new_font_size}")
    return new_font_size

def normalize_color(color):
    if isinstance(color, int):
        r = (color >> 16) & 255
        g = (color >> 8) & 255
        b = color & 255
        return (r / 255.0, g / 255.0, b / 255.0)
    elif isinstance(color, tuple) and len(color) == 3:
        return tuple(c / 255.0 if c > 1 else c for c in color)
    return (0, 0, 0)

# Fonts used in PDF
def get_fonts_used_in_pdf(doc):
    fonts_used = set()
    for page_num, page in enumerate(doc):
        font_info = extract_font_info(page)
        for info in font_info:
            fonts_used.add(info['font'])
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        fonts_used.add(span["font"])
    logging.debug(f"Fonts used in PDF: {fonts_used}")
    return fonts_used

# Determines font path based on the operating system
def register_custom_fonts_for_pdf(doc, font_dir=None):
    if font_dir is None:
        if platform.system() == "Windows":
            font_dir = Path("C:/Windows/Fonts")
        else:
            font_dir = Path(os.path.expanduser("~/.local/share/fonts"))

    if not font_dir.exists():
        logging.warning(f"Directory of font {font_dir} does not exist, i try alternative directory")
        if platform.system() == "Windows":
            font_dir = Path(os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"))
        else:
            font_dir = Path("/usr/share/fonts")
        if not font_dir.exists():
            logging.error(f"Also the alternative directory {font_dir} no exists, no custom font will be loaded")
            return {}

    logging.debug(f"Using directory of font: {font_dir}")

    fonts_used = get_fonts_used_in_pdf(doc)
    custom_fonts = {}

    for font_name in fonts_used:
        for ext in ["*.ttf", "*.otf"]:
            for font_file in font_dir.glob(ext):
                if font_name.lower() in font_file.name.lower():
                    try:
                        with open(font_file, "rb") as f:
                            font_buffer = f.read()
                        font_code = font_file.stem.lower().replace(" ", "_")
                        custom_fonts[font_code] = (font_buffer, font_name)
                        logging.debug(f"Personalized font found and registered: {font_code} per {font_name}")
                        break
                    except Exception as e:
                        logging.error(f"Error in loading {font_name} from {font_file}: {e}")

    return custom_fonts

# Calculate average color inside a rectangle (bbox), excluding text colors
def average_color(page, bbox):
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=bbox)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)
    
    font_info = extract_font_info(page)
    text_colors = [normalize_color(info['color']) for info in font_info if fitz.Rect(info['bbox']).intersects(bbox)]
    text_rgb = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in text_colors]
    
    mask = np.ones(img_array.shape[:2], dtype=bool)
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            pixel = img_array[y, x]
            if any(np.allclose(pixel, tc, atol=30) for tc in text_rgb):
                mask[y, x] = False
    
    masked_array = img_array[mask]
    if len(masked_array) == 0:
        logging.warning(f"No pixels valid for calculating the average color in the box {bbox}, use white")
        return (1, 1, 1)
    avg_color = np.mean(masked_array, axis=0) / 255.0
    logging.debug(f"Average color calculated by {bbox}: {avg_color}")
    return tuple(avg_color)

def clean_font_name(font_name):
    if not font_name:
        return "unknown", ""
    keywords = {"bold", "italic", "medium", "light", "black", "regular", "oblique", "thin", "extrabold", "extralight"}
    cleaned = re.sub(r'font|-|\d', '', font_name.lower()).strip()
    parts = cleaned.split()
    variant = next((part for part in parts if part in keywords), "")
    base_name = "".join(part for part in parts if part not in keywords and part not in {"mt", "neue"}) or parts[0] if parts else "unknown"
    return base_name, variant

def load_font_from_system(page, font_name, custom_fonts=None):
    logging.debug(f"Try to upload fonts: {font_name}")
    safe_font_name = font_name.lower().replace(" ", "_")

    if custom_fonts and safe_font_name in [name.lower().replace(" ", "_") for _, name in custom_fonts.values()]:
        for font_code, (font_buffer, orig_name) in custom_fonts.items():
            if safe_font_name == orig_name.lower().replace(" ", "_"):
                try:
                    page.insert_font(fontname=font_code, fontbuffer=font_buffer)
                    logging.debug(f"Font loaded by custom_fonts: {font_name} come {font_code}")
                    return font_code
                except Exception as e:
                    logging.error(f"Error in loading {font_name} from custom_fonts: {e}")

    base_name, variant = clean_font_name(font_name)
    logging.debug(f"Nome font base: {base_name}, Variant: {variant}")

    if platform.system() == "Windows":
        font_dirs = [
            Path("C:/Windows/Fonts"),
            Path(os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"))
        ]
    else:
        font_dirs = [
            Path(os.path.expanduser("~/.fonts")),  # Legacy
            Path(os.path.expanduser("~/.local/share/fonts")),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts")
        ]

    font_mappings = {
        "cairo": "Cairo-Regular.ttf",
        "cairolight": "Cairo-Light.ttf",
        "cairobold": "Cairo-Bold.ttf",
        "arial": "arial.ttf",
        "arialbold": "arialbd.ttf",
        "arialitalic": "ariali.ttf",
        "arialbolditalic": "arialbi.ttf",
        "arialmt": "arialmt.ttf",
        "arialboldmt": "Arial-MT-Bold.ttf",
        "arialitalicmt": "Arial-MT-Italic.otf",
        "arialmtblackitalic": "arial-mt-black-italic.ttf",
        "arialbolditalicmt": "ARIALBOLDITALICMT.OTF",
        "times": "times.ttf",
        "timesbold": "timesbd.ttf",
        "timesitalic": "timesi.ttf",
        "helvetica": "helvetic.ttf",
        "helveticalight": "Helvetica-Light.ttf",
        "helveticaneue": "HelveticaNeue-Regular.ttf",
        "helveticaneuelight": "HelveticaNeue-Light.ttf",
        "verdana": "verdana.ttf",
        "verdanabold": "verdanab.ttf",
        "verdanaitalic": "verdanai.ttf",
        "verdanabolditalic": "verdanaz.ttf",
        "tahoma": "tahoma.ttf",
        "tahomabold": "tahomabd.ttf",
        "trebuchet": "trebuc.ttf",
        "trebuchetbold": "trebucbd.ttf",
        "georgia": "georgia.ttf",
        "georgiabold": "georgiab.ttf",
        "courier": "cour.ttf",
        "courierbold": "courbd.ttf",
        "impact": "impact.ttf",
        "lucida": "lucon.ttf",
        "palatino": "pala.ttf",
        "palatinobold": "palab.ttf",
        "bookman": "bookos.ttf",
        "newcenturyschoolbook": "cschool.ttf",
        "minion": "MinionPro-Regular.otf",
        "minionbold": "MinionPro-Bold.otf",
        "lillybelle": "LillyBelle.ttf",
        "mozilla_bullet": "Mozilla_Bullet.ttf",
        "motivasanslight": "MotivaSansLight.woff.ttf",
        "motivasansthin": "MotivaSansThin.ttf",
        "motivasansregular": "MotivaSansRegular.woff.ttf",
        "motivasansbolditalic": "LillyBelle.ttf",
        "motivasansbold": "MotivaSansBold.woff.ttf",
        "minionproregular": "MinionPro-Regular.otf",
        "minionprobold": "MinionPro-Bold.otf",
        "garamond": "GARA.TTF",
        "dejavusans": "DejaVuSans.ttf",
        "dejavusansbold": "DejaVuSans-Bold.ttf",
        "dejavusansitalic": "DejaVuSans-Oblique.ttf",
        "ubuntu": "Ubuntu.ttf",
        "symbola": "Symbola.ttf"
    }

    for font_dir in font_dirs:
        if not font_dir.exists():
            continue

        logging.debug(f"Searching for fonts in the system: {font_dir}")

        for ext in ["*.ttf", "*.otf"]:
            for font_file in font_dir.glob(ext):
                file_name_lower = font_file.name.lower()
                if base_name in file_name_lower and (not variant or variant in file_name_lower):
                    try:
                        with open(font_file, "rb") as f:
                            font_buffer = f.read()
                        fontname = font_file.stem.lower().replace(" ", "_")
                        page.insert_font(fontname=fontname, fontbuffer=font_buffer)
                        logging.debug(f"Font dynamically loaded by the system: {fontname} da {font_file}")
                        return fontname
                    except Exception as e:
                        logging.error(f"Error in dynamic font loading {base_name} da {font_file}: {e}")

        for mapped_name, font_file in font_mappings.items():
            if base_name in mapped_name and (not variant or variant in mapped_name):
                full_path = font_dir / font_file
                if full_path.exists():
                    try:
                        with open(full_path, "rb") as f:
                            font_buffer = f.read()
                        fontname = mapped_name.replace(" ", "_").lower()
                        page.insert_font(fontname=fontname, fontbuffer=font_buffer)
                        logging.debug(f"Font loaded by manual mapping: {fontname} da {full_path}")
                        return fontname
                    except Exception as e:
                        logging.error(f"Error in uploading the mapped font {mapped_name} da {full_path}: {e}")

    fallback_fonts = ["DejaVuSans.ttf", "NotoSans-Regular.ttf"]
    for font_dir in font_dirs:
        for fallback_font in fallback_fonts:
            full_path = font_dir / "truetype" / "dejavu" / fallback_font if "DejaVu" in fallback_font else font_dir / fallback_font
            if full_path.exists():
                try:
                    with open(full_path, "rb") as f:
                        font_buffer = f.read()
                    fontname = fallback_font.split(".")[0].lower().replace(" ", "_")
                    page.insert_font(fontname=fontname, fontbuffer=font_buffer)
                    logging.debug(f"Font fallback loaded: {fontname} from {full_path}")
                    return fontname
                except Exception as e:
                    logging.error(f"Error in loading fallback {fallback_font}: {e}")

    logging.warning(f"No fonts found for {font_name}, last fallback set to 'helv'")
    return "helv"

# Avoid sending special characters to the translator
def split_special_chars(text):
    special_pattern = re.compile(
        r'[\x00-\x1F\x7F-\x9F\u2000-\u206F\u2190-\u21FF\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF\u2B00-\u2BFF\u3000-\u303F\uFF00-\uFFFF]'
    )
    parts = []
    last_pos = 0

    for match in special_pattern.finditer(text):
        start, end = match.span()
        if start > last_pos:
            parts.append(("text", text[last_pos:start]))
        parts.append(("special", match.group()))
        last_pos = end
    
    if last_pos < len(text):
        parts.append(("text", text[last_pos:]))
    
    return parts if parts else [("text", text)]

# Move the original text over the edges of the page using pikepdf
def remove_text(doc, temp_input, temp_pdf):
    doc.save(temp_input)
    doc.close()
    pdf = pikepdf.Pdf.open(temp_input)
    
    for page_num, page in enumerate(pdf.pages):
        try:
            media_box = page.MediaBox
            page_width = float(media_box[2]) - float(media_box[0])
            page_height = float(media_box[3]) - float(media_box[1])
            logging.debug(f"Page {page_num}: Width = {page_width}, Height = {page_height}")
            offset = max(page_width, page_height) * 2 + 2000
            logging.debug(f"Offset calculated: {offset}")

            contents = page.get("/Contents")
            if contents is None:
                logging.warning(f"Page {page_num} has no editable content")
                continue

            if isinstance(contents, pikepdf.Array):
                content_streams = [stream for stream in contents]
            else:
                content_streams = [contents]

            for i, content in enumerate(content_streams):
                try:
                    stream_data = content.read_bytes()
                    content_stream = stream_data.decode('latin-1', errors='ignore')
                    logging.debug(f"Original page content {page_num}, stream {i}: {content_stream[:100]}...")
                    new_content = []
                    lines = content_stream.splitlines()
                    modified = False
                    for line in lines:
                        line_stripped = line.strip()
                        if line_stripped.endswith("Tm"):
                            parts = line.split()
                            if len(parts) >= 6:
                                try:
                                    tx = float(parts[4]) + offset
                                    ty = float(parts[5]) + offset
                                    new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {tx} {ty} Tm"
                                    new_content.append(new_line)
                                    logging.debug(f"Modificata Tm: {line} -> {new_line}")
                                    modified = True
                                except ValueError:
                                    new_content.append(line)
                            else:
                                new_content.append(line)
                        elif line_stripped.endswith("Td"):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    tx = float(parts[0]) + offset
                                    ty = float(parts[1]) + offset
                                    new_line = f"{tx} {ty} Td"
                                    new_content.append(new_line)
                                    logging.debug(f"Modified Td: {line} -> {new_line}")
                                    modified = True
                                except ValueError:
                                    new_content.append(line)
                            else:
                                new_content.append(line)
                        elif line_stripped.endswith("TD"):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    tx = float(parts[0]) + offset
                                    ty = float(parts[1]) + offset
                                    new_line = f"{tx} {ty} TD"
                                    new_content.append(new_line)
                                    logging.debug(f"Modified TD: {line} -> {new_line}")
                                    modified = True
                                except ValueError:
                                    new_content.append(line)
                            else:
                                new_content.append(line)
                        elif line_stripped == "T*":
                            new_content.append(f"{offset} {offset} Td")
                            new_content.append("T*")
                            logging.debug(f"Modified T*: Added {offset} {offset} Td prima di T*")
                            modified = True
                        else:
                            new_content.append(line)

                    if modified:
                        new_content_stream = "\n".join(new_content).encode('latin-1')
                        content_stream_obj = pikepdf.Stream(pdf, new_content_stream)
                        if isinstance(contents, pikepdf.Array):
                            contents[i] = content_stream_obj
                        else:
                            page.Contents = content_stream_obj
                        logging.debug(f"Updated content of the page {page_num}, stream {i}")
                    else:
                        logging.debug(f"No editable text operator found in the flow {i} della pagina {page_num}")
                except Exception as e:
                    logging.error(f"Error in processing the stream {i} of the page {page_num}: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error in the page {page_num}: {e}")
            continue

    pdf.save(temp_pdf)
    pdf.close()
    logging.info(f"Original text moved to temporary file: {temp_pdf}")
	
# Removes images for text only version
def remove_text_using_pdfrw(input_pdf, output_pdf):
    input_pdf_reader = pdfrw.PdfReader(input_pdf)
    for page in input_pdf_reader.pages:
        if "/Annots" in page:
            del page["/Annots"]
        if "/Contents" in page:
            del page["/Contents"]
    writer = pdfrw.PdfWriter()
    writer.addpages(input_pdf_reader.pages)
    writer.write(output_pdf)

# Translate a PDF keeping il layout originale with PyMuPDF and pikepdf	
def translate_pdf(input_pdf, output_pdf, source_lang, target_lang, custom_font_path=None, translate_locally_path=None):
    doc = fitz.open(input_pdf)
    logging.info(f"Start extraction from PDF: {input_pdf}")
    # Register custom fonts used in PDF
    custom_fonts = register_custom_fonts_for_pdf(doc, custom_font_path) if custom_font_path else register_custom_fonts_for_pdf(doc)

    text_data = []
    for page_num, page in enumerate(doc):
        font_info = extract_font_info(page)
        for info in font_info:
            text_data.append({
                'page_num': page_num,
                'text': info['text'],
                'x0': info['bbox'][0],
                'y0': info['bbox'][1] + info['size'] * 0.8,
                'font': info['font'],
                'size': info['size'],
                'color': info['color'],
                'bbox': info['bbox']
            })

    # Check availability of translateLocally
    translate_locally_available = check_translate_locally_availability(translate_locally_path)

    available_engines = ["T", "N", "G"]
    if translate_locally_available:
        available_engines.append("L")
    # Ask the user the translation engine
    engine_prompt = f"What translation engine do you want to use between LibreTranslate (T), {'translateLocally (L), ' if 'L' in available_engines else ''}NLLB-serve (N), Google (G)? [{'/'.join(available_engines)}]: "
    engine_choice = input(engine_prompt).strip().upper()

    if engine_choice not in available_engines:
        print(f"Not valid choice! Default: LibreTranslate (T)")
        engine_choice = "T"

    # Translate the text
    for data in text_data:
        if data['text'].strip():
            parts = split_special_chars(data['text'].strip())
            translated_parts = []
            for part_type, content in parts:
                if part_type == "text":
                    translated = translate_text(content, source_lang, target_lang, engine_choice)
                    translated_parts.append(translated)
                else:
                    translated_parts.append(content)
            data['translated_text'] = "".join(translated_parts)
        else:
            data['translated_text'] = ""
    # Ask the user the translation mode
    mode_choice = input("Do you want to use overlapping rectangles (R), replace the original text (S), only text (T)? [R/S/T]: ").strip().upper()
    if mode_choice not in ["R", "S", "T"]:
        print("Not valid choice! Default: Overlay rectangles (R)")
        mode_choice = "R"
    # Create an intermediate PDF
    temp_pdf = "temp_output.pdf"
    temp_input = "temp_input.pdf"

    if mode_choice == "R":
        for page_num, page in enumerate(doc):
            font_info = extract_font_info(page)
            for info in font_info:
                x0, y0, x1, y1 = info['bbox']
                avg_color = average_color(page, info['bbox'])
                rect = fitz.Rect(x0, y0, x1, y1)
                page.draw_rect(rect, fill=avg_color, width=0)
        doc.save(temp_pdf)
        doc.close()
    elif mode_choice == "S":
        remove_text(doc, temp_input, temp_pdf)
    elif mode_choice == "T":
        doc.close()
        remove_text_using_pdfrw(input_pdf, temp_pdf)
    # Inserisci il testo tradotto con PyMuPDF
    doc = fitz.open(temp_pdf)
    for data in text_data:
        page = doc[data['page_num']]
        x0, y0 = data['x0'], data['y0']
        translated_text = data['translated_text']
        font = data['font']
        original_font_size = data['size']
        color = normalize_color(data['color'])

        font = load_font_from_system(page, font, custom_fonts=custom_fonts)
        new_font_size = calculate_font_size(data['text'], translated_text, original_font_size)

        try:
            page.insert_text(
                (x0, y0),
                translated_text,
                fontsize=new_font_size,
                color=color,
                fontname=font
            )
        except Exception as e:
            logging.warning(f"Errore con font {font}: {e}, fallback a Helvetica")
            page.insert_text(
                (x0, y0),
                translated_text,
                fontsize=new_font_size,
                color=color,
                fontname="helv"
            )

    doc.save(output_pdf)
    doc.close()
    # Pulisci i file temporanei
    os.remove(temp_pdf)
    if os.path.exists(temp_input):
        os.remove(temp_input)
    logging.info(f"PDF translated saved in: {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        logging.error("Use: python TraduciPDF.py input.pdf output.pdf language_ origin language_destination [font_customized] [translate_locally_path]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]
    source_lang = sys.argv[3]
    target_lang = sys.argv[4]
    custom_font_path = sys.argv[5] if len(sys.argv) > 5 else None
    translate_locally_path = sys.argv[6] if len(sys.argv) > 6 else None

    translate_pdf(input_pdf, output_pdf, source_lang, target_lang, custom_font_path, translate_locally_path)
