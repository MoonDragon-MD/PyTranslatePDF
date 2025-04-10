# PyTraduciPDF GUI
# V 1.0 rev3
# By MoonDragon (https://github.com/MoonDragon-MD/PyTranslatePDF)
# Dipendenze
# pip install PySimpleGUI==4.60.5.0
# avvia con
# python3 PyTraduciPDF_GUI.py

import PySimpleGUI as sg
import os
import subprocess
import sys

# Definisci il layout della GUI
sg.theme('LightGrey1')  # Tema chiaro per un aspetto pulito

layout = [
    [sg.Text('PyTraduciPDF', font=('Helvetica', 20, 'bold'), justification='center', size=(40, 1))],
    [sg.Text('Lingua di ingresso:', size=(20, 1)), sg.InputText(default_text='en', key='-SOURCE-', size=(10, 1))],
    [sg.Text('Lingua di destinazione:', size=(20, 1)), sg.InputText(default_text='it', key='-TARGET-', size=(10, 1))],
    [sg.Text('Motore di traduzione:', size=(20, 1)), 
     sg.Combo(['LibreTranslate (T)', 'translateLocally (L)', 'NLLB-serve (N)', 'Google (G)'], 
              default_value='translateLocally (L)', key='-ENGINE-', size=(20, 1))],
    [sg.Text('Opzioni PDF:', size=(20, 1)), 
     sg.Combo(['Rettangoli sovrapposti (R) - Copre il testo originale con rettangoli', 
               'Sostituisci testo originale (S) - Sposta il testo originale fuori pagina', 
               'Solo testo (T) - Rimuove tutto il resto: immagini etc'], 
              default_value='Sostituisci testo originale (S)', key='-MODE-', size=(60, 1))],
    [sg.Text('Seleziona il tuo file PDF:', size=(20, 1)), 
     sg.InputText(key='-FILE-', size=(50, 1)), 
     sg.FileBrowse(button_text='Sfoglia', file_types=(("PDF Files", "*.pdf"),), key='-BROWSE-')],
    [sg.Button('Traduci', size=(10, 1)), sg.Button('Esci', size=(10, 1))],
    [sg.Text('', key='-OUTPUT-', size=(80, 5))],  # Per mostrare messaggi all'utente
    [sg.Push(), sg.Button('Info')]
]

# Crea la finestra
window = sg.Window('PyTraduciPDF', layout, finalize=True)

# Funzione per ottenere il percorso dello script principale
def get_script_path():
    if getattr(sys, 'frozen', False):  # Se eseguito come eseguibile con PyInstaller
        return os.path.join(sys._MEIPASS, 'PyTraduciPDF.py')
    else:
        return os.path.join(os.path.dirname(__file__), 'PyTraduciPDF.py')

# Event loop
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Esci':
        break
    
    if event == 'Traduci':
        # Recupera i valori dall'interfaccia
        source_lang = values['-SOURCE-'].strip()
        target_lang = values['-TARGET-'].strip()
        engine = values['-ENGINE-'].split('(')[-1].strip(')')  # Estrai T, L, N, G
        mode = values['-MODE-'].split('(')[-1].strip(')')[:1]  # Estrai R, S, T
        input_pdf = values['-FILE-'].strip()

        # Verifica che tutti i campi siano compilati
        if not all([source_lang, target_lang, engine, mode, input_pdf]):
            window['-OUTPUT-'].update('Errore: Compila tutti i campi!', text_color='red')
            continue

        if not os.path.exists(input_pdf):
            window['-OUTPUT-'].update('Errore: File PDF non trovato!', text_color='red')
            continue

        # Costruisci il nome del file di output
        input_dir = os.path.dirname(input_pdf)
        input_name = os.path.splitext(os.path.basename(input_pdf))[0]
        output_pdf = os.path.join(input_dir, f"{input_name}_tradotto.pdf")

        # Costruisci il comando per chiamare PyTraduciPDF.py
        script_path = get_script_path()
        command = [
            sys.executable,  # Usa l'interprete Python corrente
            script_path,
            input_pdf,
            output_pdf,
            source_lang,
            target_lang
        ]

        # Passa i valori come input simulando l'interazione con l'utente
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Invia le scelte per motore e modalità come input
            inputs = f"{engine}\n{mode}\n"
            stdout, stderr = process.communicate(input=inputs)

            if process.returncode == 0:
                window['-OUTPUT-'].update(f'Traduzione completata: {output_pdf}', text_color='green')
            else:
                window['-OUTPUT-'].update(f'Errore durante la traduzione:\n{stderr}', text_color='red')
        except Exception as e:
            window['-OUTPUT-'].update(f'Errore: {str(e)}', text_color='red')

    # Controlla se l'utente ha cliccato sul pulsante Info
    if event == 'Info':
        # finestra delle informazioni
        info_layout = [
            [sg.Text('PyTraduciPDF')],
            [sg.Text('Versione 1.0')],
			[sg.Text('Creata da MoonDragon-MD')],
            [sg.Text("Sito Web: "), sg.InputText("https://github.com/MoonDragon-MD/PyTranslatePDF", readonly=True)],
            [sg.Text('Funziona con vari traduttori offline + Google online')],
            [sg.Text("Dipendenze: "), sg.InputText("pip install PySimpleGUI==4.60.5.0 PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf", readonly=True)],
            [sg.Text('Dipendenze opzionali, traduttori offline:')],			
            [sg.Text("translateLocally: "), sg.InputText("https://github.com/XapaJIaMnu/translateLocally", readonly=True)],
            [sg.Text("LibreTranslate: "), sg.InputText("https://github.com/LibreTranslate/LibreTranslate", readonly=True)],
            [sg.Text("nllb-serve: "), sg.InputText("https://github.com/thammegowda/nllb-serve", readonly=True)],
            [sg.Button('Chiudi')]
        ]
        
        # Crea la finestra delle informazioni
        info_window = sg.Window('Informazioni', info_layout)
        
        # Ciclo per gestire gli eventi della finestra delle informazioni
        while True:
            info_event, info_values = info_window.read()
            
            # Controlla se l'utente ha chiuso la finestra delle informazioni
            if info_event == sg.WINDOW_CLOSED or info_event == 'Chiudi':
                break
        
        # Chiudi la finestra delle informazioni
        info_window.close()
		
window.close()
