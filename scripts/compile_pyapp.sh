#!/bin/bash

set -e  # Stop on error
set -x # Echo commands

# Assert uv, rustup and docker are installed
for cmd in uv rustup docker; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "$cmd could not be found, please install it first."
    exit 1
  fi
done

# Assert script is run on a linux amd64 host
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This script must be run on a Linux amd64 host."
  exit 1
fi

# Variables
PROJECT_NAME="glados"
BINARY_NAME="GLaDOS"
VERSION="0.1.0"
EXE_BASE="${BINARY_NAME}-${VERSION}"
WHEEL_PATH="$(pwd)/dist/${PROJECT_NAME}-${VERSION}-py3-none-any.whl"

# Export PYAPP variables
export PYAPP_PROJECT_NAME="${PROJECT_NAME}"
export PYAPP_PROJECT_VERSION="${VERSION}"
export PYAPP_PROJECT_PATH="${WHEEL_PATH}"
export PYAPP_EXEC_MODULE="${PROJECT_NAME}"
export PYAPP_PYTHON_VERSION="3.12"
export PYAPP_DISTRIBUTION_EMBED=1

# Build wheel
uv sync
uv build --wheel

# Ensure cross-rs is installed
cargo install cross --git https://github.com/cross-rs/cross

# Download pyapp source if not present
if [ ! -d "pyapp-latest" ]; then
  curl -L https://github.com/ofek/pyapp/releases/latest/download/source.tar.gz | tar -xz
  mv pyapp-v* pyapp-latest

  # Here we create a Cross.toml config where we say that we want to mount the wheel file
  # inside the container that cross-rs will create, this is needed because during the 
  # compilation of pyapp, it uses the wheel file to create the final binary.
  # The way it works is that we give the name of the environnement variable that cross-rs
  # will evaluate to create the volume
  cat <<EOF > "pyapp-latest/Cross.toml"
[build.env]
volumes = ["PYAPP_PROJECT_PATH"]
EOF
fi

# Cross-compile targets: os arch target ext
targets=(
  # We choose gnu because musl doesn't work with onnxruntime-gpu
  "x86_64-pc-windows-gnu"
  "x86_64-unknown-linux-gnu"

  # I did not manage to get this working :(
  # "aarch64-unknown-linux-gnu"
)

# Build for each target if binary not present
for target in "${targets[@]}"; do
  # Split target into array by '-'
  IFS='-' read -r arch _ os _ <<< "$target"

  # Determine extension
  if [[ "$os" == "windows" ]]; then
    ext=".exe"
  else
    ext=""
  fi

  # Install missing Rust targets (necessary ?)
  if ! rustup target list --installed | grep -q "^${target}$"; then
    rustup target add "${target}"
  fi

  binary_path="pyapp-latest/target/${target}/release/pyapp${ext}"  # Cargo names it pyapp or pyapp.exe
  if [ ! -f "${binary_path}" ]; then
    pushd pyapp-latest
    CROSS_USE_SYSTEM_LIBS=0 cross build --release --target "${target}"
    popd
  fi
  cp "${binary_path}" "dist/${EXE_BASE}-${os}-${arch}${ext}"
done

# Download appimagetool if not present
appimagetool_file="appimagetool-x86_64.AppImage"
if [ ! -f "dist/${appimagetool_file}" ]; then
  curl -L "https://github.com/AppImage/appimagetool/releases/download/continuous/${appimagetool_file}" -o "dist/${appimagetool_file}"
fi
chmod +x "dist/${appimagetool_file}"

# Create AppDir
APPDIR="dist/glados.AppDir"
rm -rf "${APPDIR}" # Clean up previous builds to avoid potential issues
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps/"
mkdir -p "${APPDIR}/usr/share/metainfo/"

# Copy binary
cp "dist/${BINARY_NAME}-${VERSION}-linux-x86_64" "${APPDIR}/usr/bin/${BINARY_NAME}"

# Desktop file
cat <<EOF > "${APPDIR}/glados.desktop"
[Desktop Entry]
Type=Application
Name=${BINARY_NAME}
X-AppImage-Version=${VERSION}
Comment=Genetic Lifeform and Disk Operating System
Exec=AppRun
Icon=aperture_icon
Categories=Utility;
Terminal=false
EOF

# AppData metadata
cat <<EOF > "${APPDIR}/usr/share/metainfo/io.github.glados.metainfo.xml"
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
<id>io.github.glados</id>
<name>GLaDOS</name>
<summary>Genetic Lifeform and Disk Operating System</summary>
<component type="desktop-application" license="MIT">
<content_rating type="oars-1.1"/>
<developer_name>dnhkng</developer_name>
<url type="homepage">https://github.com/dnhkng/GLaDOS</url>
<description>
  <p>
    This is the AI system of a project dedicated to building a real-life version of GLaDOS.
    One of the initial objectives is to have a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.
    The ultimate goal is to have an aware, interactive, and embodied GLaDOS.
  </p>
</description>
<launchable type="desktop-id">glados.desktop</launchable>
<releases>
  <release version="${VERSION}" date="$(date +%Y-%m-%d)"/>
</releases>
</component>
EOF

# Icons (we need to copy it 2 times in order for it to work properly)
cp images/aperture_icon.png "${APPDIR}/aperture_icon.png"
cp images/aperture_icon.png "${APPDIR}/usr/share/icons/hicolor/256x256/apps/aperture_icon.png"

# AppRun
cat <<'EOF' > "${APPDIR}/AppRun"
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PYAPP_RUNNING=1
export PYAPP_RELATIVE_DIR="$(pwd)"
x-terminal-emulator -e "$HERE/usr/bin/GLaDOS" "$@"
EOF
chmod +x "${APPDIR}/AppRun"

# Build AppImage
"./dist/${appimagetool_file}" "${APPDIR}" "dist/GLaDOS-${VERSION}-linux-x86_64.AppImage"
