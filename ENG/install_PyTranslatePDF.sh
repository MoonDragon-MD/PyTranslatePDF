#!/bin/bash

# Show the main installer window
(
    while true; do
        sleep 1
    done
) | zenity --progress --width=600 --height=500 --title="Installatore per PyTranslatePDF by MoonDragon" \
    --text="<b>Installator for PyTranslatePDF by MoonDragon</b>\n\nVersions: 1.0.0\n\nhttps://github.com/MoonDragon-MD/PyTranslatePDF\n\nFollow the guided installation including dependencies and shortcuts on the menu" \
    --no-cancel --auto-close --pulsate &

INSTALLER_PID=$!

# Function to show a popup with the command to run
show_command_popup() {
    zenity --error --width=400 --text="Errors: $1 not found.\nExecute the following command:\n\n<b>$2</b>"
}

# Check dependencies
if ! zenity --question --width=400 --text="Want to check and install dependencies?"; then
    INSTALL_DEPENDENCIES=false
else
    INSTALL_DEPENDENCIES=true
fi

if [ "$INSTALL_DEPENDENCIES" = true ]; then
    # Check Python3
    if ! command -v python3 &> /dev/null; then
        show_command_popup "Python3" "sudo apt-get install python3"
        kill $INSTALLER_PID
        exit 1
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        show_command_popup "pip3" "sudo apt-get install python3-pip"
        kill $INSTALLER_PID
        exit 1
    fi

    # Check fonts-dejavu
    if ! dpkg-query -W fonts-dejavu &>/dev/null; then
        show_command_popup "fonts-dejavu" "sudo apt-get install fonts-dejavu"
        kill $INSTALLER_PID
        exit 1
    fi

    # Install Python dependencies
    zenity --info --width=400 --text="Installing Python dependencies..."
    pip3 install PyMuPDF requests googletrans==3.1.0a0 numpy Pillow pdfrw pikepdf
fi

# Asks the user where to install PyTranslatePDF
INSTALL_DIR=$(zenity --file-selection --directory --title="Select the installation directory for PyTranslatePDF" --width=400)
if [ -z "$INSTALL_DIR" ]; then
    zenity --error --width=400 --text="No selected directories.\nInstallation cancelled."
    kill $INSTALLER_PID
    exit 1
fi

# Make desktop entry
zenity --info --width=400 --text="Creating the link in the application menu..."
cat > ~/.local/share/applications/PyTranslatePDF.desktop << EOL
[Desktop Entry]
Name=PyTranslatePDF
Comment=PDF translator
Exec=$INSTALL_DIR/PyTranslatePDF/StartUbuntu.sh
Icon=$INSTALL_DIR/PyTranslatePDF/icon.png
Terminal=false
Type=Application
Categories=Utility;Office;
EOL

# Make the installation directory if there is no
mkdir -p "$INSTALL_DIR"

# Copy the required files
zenity --info --width=400 --text="Installing the application..."
cp -r PyTranslatePDF "$INSTALL_DIR/"

# Generate the script StartUbuntu.sh
cat > "$INSTALL_DIR/PyTranslatePDF/StartUbuntu.sh" << EOL
#!/bin/bash
cd $INSTALL_DIR/PyTranslatePDF/
python3 PyTraduciPDF_GUI.py
EOL

# Makes the script executable StartUbuntu.sh
chmod +x "$INSTALL_DIR/PyTranslatePDF/StartUbuntu.sh"

# Close the main installer window
kill $INSTALLER_PID

zenity --info --width=400 --text="Complete installation!"
zenity --info --width=400 --text="You can start PyTranslatePDF from the application menu or running 'PyTranslatePDF' in the terminal"
