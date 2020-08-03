#!/usr/bin/env bash

# ================== #
# Tim de Klijn, 2020 #
# download_model.sh  #
# ================== #

# Check if the MODELDIR exists, if it does, remove it.
# Then create the directory and download the model in this

echo "===================================================="

# Set model dir
MODELDIR="tst_model"

# Delete the directory if it exists
echo "Deleting the model directory if it exists"
if [ -d "$MODELDIR" ]; then
  rm -Rf $MODELDIR
fi

# Create the model directory
echo "Creating the model directory"
mkdir -p "$MODELDIR/1"

# Download the model
echo "Downloading the model"
wget -O tmp.tar https://tfhub.dev/see--/bert-uncased-tf2-qa/1?tf-hub-format=compressed

# Extracting the model and cleaning up the tmp file
echo "Extracting model"
tar -xzvf tmp.tar -C tst_model/1
echo "Removing model tar file"
rm tmp.tar

echo "Model has been downloaded"
