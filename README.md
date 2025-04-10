# PyTranslatePDF
Program to translate PDF while maintaining the layout and fonts. You can use several offline translation engines ( LibreTranslate, translateLocally, NLLB-serve ) + Google

Works on both GNU/Linux (Ex:Ubuntu/Debian) and Windows

## Description
I created this program to translate mainly offline PDFs without losing the layout, font, font color, original images.

It was not easy and maybe still not perfect.

One very important thing to keep the layout is fonts: you have to have the PDF fonts in the system to have a proper layout of the text. I put an automatic search method but it doesn't always succeed, so I added a mapping for fonts, here you can add your own (you can help by enabling loggin).

I set that special carraters don't go to the translator because they can give problems.

There are three modes of editing 

- translated text only

- replacing original text with translated

- overlay via rectangles with translated text on top.

The latter is needed in special cases where there is writing both on the images and on top of them, or in special cases where using substitution you can no longer see anything (depending on how they created the PDF)

For camouflage I set that the color of the rectangle is calculated based on the average of the background colors under that text box.

Here is an example:

![alt text](https://github.com/MoonDragon-MD/PyTranslatePDF/blob/main/img/Comparazione.png?raw=true)

## Dependencies
- Python installed on the system, as a minimum python version 3.8
- ```sudo apt install fonts-dejavu```       (with Linux)
- ```pip install PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf```

## Optional dependencies
- [translateLocally](https://github.com/XapaJIaMnu/translateLocally) If you want a fast and easy-to-use engine without gigs of dependencies - Low level of translation [Here are the additional repositories to get more languages (https://raw.githubusercontent.com/hplt-project/bitextor-mt-models/refs/heads/main/models.json ; https://object.pouta.csc.fi/OPUS-MT-models/app/models.json ; https://translatelocally.com/models.json ) Those who want Italian [look here](https://github.com/MoonDragon-MD/ITA-models-translateLocally-) ]
- [LibreTranslate](https://github.com/XapaJIaMnu/translateLocally) Very fast (recommended) - Average level of translation
- [NLLB-serve](https://github.com/thammegowda/nllb-serve) If you want the CPU-only version look at [my fork](https://github.com/MoonDragon-MD/nllb-serve-slim) - Medium/high translation level , low speed
- ```pip install pymupdf-fonts```       (To increase the available fonts without going to change the font mapping)

## Installation
GNU/Linux     ```./install_PyTranslatePDF.sh``` (Adds shortcut with icon to main menu)

## Usage
Portable GNU/Linux            ```StartUbuntu.sh``` (double-click and then click run) or in terminal ```./StartUbuntu.sh```

Portable Windows              ```StartWindows.bat``` (double-click)

Direct start GUI GNU/Linux    ```python3 PyTranslatePDF_GUI.py```

Direct start GNU/Linux        ```python3 PyTranslatePDF.py name.pdf name_translate.pdf en it [font_customized] [translate_locally_path]``` (these last two are optional)

Direct start GUI Windows      ```python PyTranslatePDF_GUI.py```

Direct start Windows          ```python PyTranslatePDF.py name.pdf name_translate.pdf en it [font_customized] [translate_locally_path]``` (these last two are optional)

## Screenshot
![alt text](https://github.com/MoonDragon-MD/PyTranslatePDF/blob/main/img/Schermata.jpg?raw=true)
