#!/bin/bash

# Check if the virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment already activated."
fi

python app.py
