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

if [ -z ${SS_ENV} ]; then
    SS_ENV="dev"
    export FLASK_DEBUG=1
elif [ "${SS_ENV}" = "dev" ]; then
    export FLASK_DEBUG=1
fi

export FLASK_ENV=${SS_ENV}
export GRAPH_ENV=${SS_ENV}

FLASK_APP=smart_summary.py flask run --host 0.0.0.0 --port ${PORT}
