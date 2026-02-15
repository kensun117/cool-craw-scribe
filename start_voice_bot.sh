#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export $(grep -v '^#' .env.voice | xargs)
python voice_bot.py
