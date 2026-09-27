#!/bin/bash

OUTPUT="structure.txt"

{
    echo "APPLICATION STRUCTURE"
    echo "====================="
    echo ""

    find . \
        -not -path './.git*' \
        -not -path './.venv*' \
        -not -path './venv*' \
        -not -path './__pycache__*' \
        -not -path './node_modules*' \
        -not -name "$OUTPUT" \
        | sort
} > "$OUTPUT"

echo "Created $OUTPUT"