# ESP32 Source-Code Corpus

This directory holds the **ESP32 source code** used to build the code RAG
index. By default the code build reads from here (`code.docs_dir`, default
`source_code`).

## How to use

1. Place (or clone) the source code in this directory.
2. Build the **code** index (opt-in, separate database):

   ```bash
   # Windows PowerShell
   $env:RAG_INDEX_CODE="1"; python -m scripts.build.run

   # Linux / macOS
   RAG_INDEX_CODE=1 python -m scripts.build.run
   ```

   This writes the code vector DB to `.chroma_esp32_code/`.
3. Query the code index:

   ```bash
   python scripts/agent.py "bootloader partition table" --code
   ```

## Download the files

Get the source-code files from GitHub:

https://github.com/yezeganghelei/ESP32

The ESP32 repository is a **single tree**. `./setup.sh --corpus` (or
`.\setup.ps1 --corpus`) downloads it and moves `01. Software Code` here
automatically (the other top-level directories go to `source/`).

## Current contents

- `01. Software Code/`
