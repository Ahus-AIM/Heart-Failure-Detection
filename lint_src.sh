#!/bin/bash
set -e

echo "Running black..."
black src

echo "Running isort..."
isort src

echo "Running flake8..."
flake8 src
