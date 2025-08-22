#!/bin/bash
# Install SC2 and add the custom maps

# Clone the source code.
#git clone git@github.com:tjuHaoXiaotian/pymarl3.git
# Run this script in the root directory of the SMAC project
export CODE_DIR=$(pwd)

# 1. Install StarCraftII
echo 'Install StarCraftII...'
cd "$CODE_DIR/3rdparty"
export SC2PATH="$CODE_DIR/3rdparty"

echo 'SC2PATH is set to '$SC2PATH
if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
	mkdir -p "$SC2PATH"
        unzip -P iagreetotheeula SC2.4.10.zip -d "$SC2PATH"
else
        echo 'StarCraftII is already installed.'
fi

# Remove the StarCraft zip file after installation.
rm -rf "$CODE_DIR/SC2.4.10.zip"

# 2. Install the custom maps

# Copy the maps to the target dir.
echo 'Install SMACV1 and SMACV2 maps...'
MAP_DIR="$SC2PATH/StarCraftII/Maps/"
if [ ! -d "$MAP_DIR/SMAC_Maps" ]; then
    echo 'MAP_DIR is set to '$MAP_DIR
    if [ ! -d $MAP_DIR ]; then
            mkdir -p $MAP_DIR
    fi
    cp -r "$CODE_DIR/src/envs/smac_v2/official/maps/SMAC_Maps" $MAP_DIR
else
    echo 'SMACV1 and SMACV2 maps are already installed.'
fi
echo 'StarCraft II and SMAC maps are installed.'
