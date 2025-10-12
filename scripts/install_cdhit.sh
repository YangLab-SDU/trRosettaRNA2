#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e

echo -e "${GREEN}Downloading CD-HIT...${NC}"

SOURCE_URL="https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cdhit/cd-hit-v4.5.5-2011-03-31.tgz"
SOURCE_FILE=$(basename "$SOURCE_URL")
SOURCE_DIR="cd-hit-v4.5.5-2011-03-31"

if [ ! -f "$SOURCE_FILE" ]; then
    wget "$SOURCE_URL"
else
    echo "'$SOURCE_FILE' already exists. Skip downloads."
fi

echo -e "${GREEN}Compling CD-HIT...${NC}"

if [ -d "$SOURCE_DIR" ]; then
    rm -rf "$SOURCE_DIR"
fi
tar -xvf "$SOURCE_FILE"

cd "$SOURCE_DIR"

TARGET_FILE="cdhit-common.h"
OLD_CODE="push_back( item );"
NEW_CODE="this->push_back( item );"

sed -i.bak "s/${OLD_CODE}/${NEW_CODE}/g" "$TARGET_FILE"

make
echo -e "\n${GREEN}CD-HIT installed!${NC}"

cd ..