#!/usr/bin/env bash

PORT=5000

which python 
if [ $? -ne 0 ]; then
    echo "python not found"
    exit 1
else
    py_ver="$(python -V 2>&1)"
    if [[ ${py_ver} != *"2.7"* ]]; then
        echo "${py_ver} found. Expected python2.7 "
        exit 1
    fi
fi

FLASK_DEBUG=1 FLASK_APP=smart_summary.py flask run --host 0.0.0.0 --port ${PORT}
