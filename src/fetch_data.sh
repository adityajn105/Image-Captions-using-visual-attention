#!/bin/bash
FILE="kaggle.json"
if [ -f  "$FILE" ];
then
	mkdir ~/.kaggle
	mv kaggle.json ~/.kaggle/

	kaggle datasets download -d hsankesara/flickr-image-dataset
	unzip -q flickr-image-dataset.zip
	mv flickr30k_images/results.csv .

	kaggle datasets download -d watts2/glove6b50dtxt
	unzip -q glove6b50dtxt.zip
	rm -r *.zip 
else
	echo "File $FILE does not exist" >&2
fi