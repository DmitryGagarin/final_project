#!/bin/bash

find . -type f -name "*.py" -print0 | sort -z | while IFS= read -r -d '' file; do
    echo "================================================================================" >> all_code.txt
    echo "FILE: $file" >> all_code.txt
    echo "================================================================================" >> all_code.txt
    echo "" >> all_code.txt
    cat "$file" >> all_code.txt
    echo -e "\n" >> all_code.txt
done