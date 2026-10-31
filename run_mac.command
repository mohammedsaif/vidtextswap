#!/bin/bash
cd "$(dirname "$0")"
echo "Checking requirements (the first run takes a few minutes)..."
python3 -m pip install -q -r requirements.txt || { echo "Install Python 3.10+ from python.org first."; exit 1; }
python3 -m vidtextswap "$@"
