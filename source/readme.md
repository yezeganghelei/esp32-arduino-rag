# ESP32 Documentation Corpus

This directory holds the raw **ESP32 documentation** used to build the
documentation RAG index. By default the build reads from here
(`RAG_DOCS_DIR`, default `<skill_dir>/source`).

## How to use

1. Place (or clone) the documentation files in this directory. Supported
   formats: `.pdf`, `.zip`, `.xlsx`, `.docx`, `.md`, `.markdown`.
2. Build the index:

   ```bash
   python -m scripts.build.run
   ```

   This writes the vector DB to `.chroma_esp32_all/` (incremental on re-run).

## Download the files

Get the documentation files from GitHub:

https://github.com/yezeganghelei/ESP32

The ESP32 repository is a **single tree**. The fastest way to populate both
directories is `./setup.sh --corpus` (or `.\setup.ps1 --corpus`), which splits
it automatically: `01. Software Code` goes to `source_code/`, every other
top-level directory goes here. To do it by hand, move `01. Software Code` to
`source_code/` and everything else into this directory.

## Current contents

- `02. Hardware Documents/`
- `03. ESP32-Tutorials/`
- `04. ESP32 Series Reference Resource/`
- `06. ESP32-S3 Camera Reference Resource/`
- `07. Breadboard Power Supply Board/`
- `09. Arduino-UNO-Tutorial/`
