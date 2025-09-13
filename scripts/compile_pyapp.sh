#!/bin/bash

# TODO
# - Cleaner
# - Stop the script on error
# - What else could I add as metadata for the Appimage ? Appdetails ?

uv sync
uv build --wheel

if [ ! -d "pyapp-latest" ]; then
  # Download pyapp and untar it directly, so it doesn't create a file
  curl -L https://github.com/ofek/pyapp/releases/latest/download/source.tar.gz | tar -xz
  mv pyapp-v* pyapp-latest
fi

# Build the project into an executable
export PYAPP_PROJECT_NAME="glados"
export PYAPP_PROJECT_VERSION="0.1.0"
export PYAPP_PROJECT_PATH=../dist/glados-0.1.0-py3-none-any.whl
export PYAPP_EXEC_MODULE="glados"
export PYAPP_PYTHON_VERSION="3.12"
export PYAPP_DISTRIBUTION_EMBED=1
pushd pyapp-latest
cargo build --release
popd
export EXE_NAME="$PYAPP_PROJECT_NAME-$PYAPP_PROJECT_VERSION"
mv pyapp-latest/target/release/pyapp "dist/$EXE_NAME"
"dist/$EXE_NAME" self remove # Just so we are sure there is no cache anywhere

# Create AppDir structure so that appimagetool can package it into an AppImage
export APPDIR=dist/glados.AppDir
mkdir -p $APPDIR/usr/bin
mkdir -p $APPDIR/usr/share/icons/hicolor/256x256/apps/
cp "dist/$EXE_NAME" "$APPDIR/usr/bin/$EXE_NAME"
echo "[Desktop Entry]
Type=Application
Name=GLaDOS v$PYAPP_PROJECT_VERSION
Comment=GLaDOS - Genetic Lifeform and Disk Operating System
Exec=AppRun
Icon=aperture_icon
Categories=Utility;" > $APPDIR/glados.desktop
cp images/aperture_icon.png $APPDIR/aperture_icon.png
cp images/aperture_icon.png $APPDIR/usr/share/icons/hicolor/256x256/apps/aperture_icon.png
echo "#!/bin/sh
HERE=\"\$(dirname \"\$(readlink -f \"\$0\")\")\"
export PYAPP_RUNNING=1
export PYAPP_RELATIVE_DIR=\$(pwd)
x-terminal-emulator -e \"\$HERE/usr/bin/$EXE_NAME\" \"\$@\"" > $APPDIR/AppRun
chmod +x $APPDIR/AppRun
curl -L https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage -o dist/appimagetool
chmod +x dist/appimagetool
./dist/appimagetool $APPDIR
rm -rf $APPDIR # Needed to clean up, otherwise stuff may appear on next build
mv GLaDOS*.AppImage "dist/GLaDOS-$PYAPP_PROJECT_VERSION-x86_64.AppImage"
