# PyTranslatePDF GUI
# V 1.0 rev3
# By MoonDragon (https://github.com/MoonDragon-MD/PyTranslatePDF)
# Dipendence
# pip install PySimpleGUI==4.60.5.0
# start with
# python3 PyTranslatePDF_GUI.py

import PySimpleGUI as sg
import os
import subprocess
import sys

# Define the GUI layout
sg.theme('LightGrey1')  # Clear theme for a clean look

layout = [
    [sg.Text('PyTranslatePDF', font=('Helvetica', 20, 'bold'), justification='center', size=(40, 1))],
    [sg.Text('Source language:', size=(20, 1)), sg.InputText(default_text='it', key='-SOURCE-', size=(10, 1))],
    [sg.Text('Destination language:', size=(20, 1)), sg.InputText(default_text='en', key='-TARGET-', size=(10, 1))],
    [sg.Text('Translator engine:', size=(20, 1)), 
     sg.Combo(['LibreTranslate (T)', 'translateLocally (L)', 'NLLB-serve (N)', 'Google (G)'], 
              default_value='translateLocally (L)', key='-ENGINE-', size=(20, 1))],
    [sg.Text('Option PDF:', size=(20, 1)), 
     sg.Combo(['Overlay rectangles (R) - Cover the original text with rectangles', 
               'Replace original text (S) - Move the original text out of page', 
               'Only text (T) - Removes everything else: pictures etc'], 
              default_value='Replace original texts (S)', key='-MODE-', size=(60, 1))],
    [sg.Text('Select your PDF file:', size=(20, 1)), 
     sg.InputText(key='-FILE-', size=(50, 1)), 
     sg.FileBrowse(button_text='Browse', file_types=(("PDF Files", "*.pdf"),), key='-BROWSE-')],
    [sg.Button('Translate', size=(10, 1)), sg.Button('Exit', size=(10, 1))],
    [sg.Text('', key='-OUTPUT-', size=(80, 5))],  # To show user messages
    [sg.Push(), sg.Button('Info')]
]

# Make window
window = sg.Window('PyTranslatePDF', layout, finalize=True)

# Function to get the main script path
def get_script_path():
    if getattr(sys, 'frozen', False):  # If executed as executable with PyInstaller
        return os.path.join(sys._MEIPASS, 'PyTranslatePDF.py')
    else:
        return os.path.join(os.path.dirname(__file__), 'PyTranslatePDF.py')

# Event loop
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    if event == 'Translate':
        # Recovers values from the interface
        source_lang = values['-SOURCE-'].strip()
        target_lang = values['-TARGET-'].strip()
        engine = values['-ENGINE-'].split('(')[-1].strip(')')  # Extract T, L, N, G
        mode = values['-MODE-'].split('(')[-1].strip(')')[:1]  # Extract R, S, T
        input_pdf = values['-FILE-'].strip()

        # Check that all box are compiled
        if not all([source_lang, target_lang, engine, mode, input_pdf]):
            window['-OUTPUT-'].update('Error: Fill all box!', text_color='red')
            continue

        if not os.path.exists(input_pdf):
            window['-OUTPUT-'].update('Error: PDF file not found!', text_color='red')
            continue

        # Build the output file name
        input_dir = os.path.dirname(input_pdf)
        input_name = os.path.splitext(os.path.basename(input_pdf))[0]
        output_pdf = os.path.join(input_dir, f"{input_name}_translated.pdf")

        # Build command to call PyTranslatePDF.py
        script_path = get_script_path()
        command = [
            sys.executable,  # Use current Python interpreter
            script_path,
            input_pdf,
            output_pdf,
            source_lang,
            target_lang
        ]

        # Switch values as input by simulating user interaction
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Send choices for motor and mode as input
            inputs = f"{engine}\n{mode}\n"
            stdout, stderr = process.communicate(input=inputs)

            if process.returncode == 0:
                window['-OUTPUT-'].update(f'Complete translation: {output_pdf}', text_color='green')
            else:
                window['-OUTPUT-'].update(f'Error during translation:\n{stderr}', text_color='red')
        except Exception as e:
            window['-OUTPUT-'].update(f'Errore: {str(e)}', text_color='red')

    # Check if you clicked on the Info button
    if event == 'Info':
        # information window
        info_layout = [
            [sg.Text('PyTranslatePDF')],
            [sg.Text('Versione 1.0')],
			[sg.Text('Created by MoonDragon-MD')],
            [sg.Text("Web-site: "), sg.InputText("https://github.com/MoonDragon-MD/PyTranslatePDF", readonly=True)],
            [sg.Text('Works with various translators offline + Google online')],
            [sg.Text("Dipendence: "), sg.InputText("pip install PySimpleGUI==4.60.5.0 PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf", readonly=True)],
            [sg.Text('Optional dipendence, offline translators:')],			
            [sg.Text("translateLocally: "), sg.InputText("https://github.com/XapaJIaMnu/translateLocally", readonly=True)],
            [sg.Text("LibreTranslate: "), sg.InputText("https://github.com/LibreTranslate/LibreTranslate", readonly=True)],
            [sg.Text("nllb-serve: "), sg.InputText("https://github.com/thammegowda/nllb-serve", readonly=True)],
            [sg.Button('Close')]
        ]
        
        # Create the information window
        info_window = sg.Window('Information', info_layout)
        
        # Cycle to manage information window events
        while True:
            info_event, info_values = info_window.read()
            
            # Check if the user has closed the information window
            if info_event == sg.WINDOW_CLOSED or info_event == 'Close':
                break
        
        # Close the information window
        info_window.close()
		
window.close()
