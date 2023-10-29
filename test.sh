#!/bin/sh

MODEL=$1

if [ "$MODEL" = "FULL" ] || [ "$MODEL" = "STF-None" ]; then
    python ./src/demo.py
elif [ "$MODEL" = "ITC" ]; then
    python ./src/demo_ITC.py
else
    echo "Unknown model. Please specify FULL, STF-None, or ITC."
fi

