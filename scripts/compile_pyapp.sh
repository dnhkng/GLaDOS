#!/bin/bash

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
cd pyapp-latest
cargo build --release
cd ..
mv pyapp-latest/target/release/pyapp dist/glados
dist/glados self remove # Just so we are sure there is no cache anywhere
