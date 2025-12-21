#!/bin/bash
set -e

# Define variables
APP_NAME="MeasureLab"
APP_DIR="AppDir"
LINUXDEPLOY="linuxdeploy-x86_64.AppImage"

# Clean up previous build
rm -rf $APP_DIR $APP_NAME*.AppImage

# Download linuxdeploy if not present
if [ ! -f "$LINUXDEPLOY" ]; then
    wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
    chmod +x $LINUXDEPLOY
fi

# Create AppDir structure
mkdir -p $APP_DIR/usr/bin
mkdir -p $APP_DIR/usr/share/icons/hicolor/256x256/apps
mkdir -p $APP_DIR/usr/share/applications

# Copy application binary
cp dist/MeasureLab $APP_DIR/usr/bin/

# Copy icon and desktop file
cp app_icon.png $APP_DIR/usr/share/icons/hicolor/256x256/apps/app_icon.png
cp audio-tools.desktop $APP_DIR/usr/share/applications/

# Update desktop file Exec path
sed -i 's/Exec=main_gui/Exec=MeasureLab/g' $APP_DIR/usr/share/applications/audio-tools.desktop

# Initialize AppDir with linuxdeploy
./$LINUXDEPLOY --appdir $APP_DIR --output appimage \
    --desktop-file $APP_DIR/usr/share/applications/audio-tools.desktop \
    --icon-file $APP_DIR/usr/share/icons/hicolor/256x256/apps/app_icon.png \
    --executable $APP_DIR/usr/bin/MeasureLab

echo "AppImage build complete!"
