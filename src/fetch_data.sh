#!/bin/bash
TYPE="8k"
FILE="kaggle.json"

if [ $# -ge 1 ];
then
	TYPE=$1
fi

if [ -f  "$FILE" ];
then
	mkdir ~/.kaggle
	mv kaggle.json ~/.kaggle/
	
	kaggle datasets download -d adityajn105/glove6b50d
	unzip -q glove6b50d.zip
	rm -r glove6b50d.zip
else
	echo "File $FILE does not exist" >&2
	exit;
fi


if [ $TYPE == "8k" ];
then
	kaggle datasets download -d adityajn105/flickr8k
	unzip -q flickr8k.zip
	rm -r flickr8k.zip
else
	kaggle datasets download -d adityajn105/flickr30k
	unzip -q flickr30k.zip
	rm -r flickr30k.zip
fi
