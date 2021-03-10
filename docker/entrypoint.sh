#!/bin/bash

VOLUME=volume
SCRIPT=scripts/$1

if [[ ! -f "$SCRIPT" ]]; then
  echo File not found: "$SCRIPT"
  exit 1
fi

if [[ ! -d "$VOLUME" ]]; then
  echo Folder not found: "$VOLUME"
  exit 1
fi

cd "$VOLUME" || exit 1
SCRIPT=../"$SCRIPT"

# Call script and forward the rest of the parameters given to docker

python "$SCRIPT" "${@:2}"
