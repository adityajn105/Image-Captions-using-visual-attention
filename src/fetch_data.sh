#!/bin/bash
FILE="kaggle.json"
if [ -f  "$FILE" ];
then
	mkdir ~/.kaggle
	mv kaggle.json ~/.kaggle/

	kaggle datasets download -d shadabhussain/flickr8k
	unzip -q flickr8k.zip
	rm -r Flickr_Data
	mv flickr_data/Flickr_Data .
	mv Flickr_Data/Images .
	mv Flickr_Data/Flickr_TextData/Flickr8k.token.txt captions.txt
	rm -r flickr_data Flickr_Data flickr8k.zip

	kaggle datasets download -d watts2/glove6b50dtxt
	unzip -q glove6b50dtxt.zip
	rm -r *.zip 
else
	echo "File $FILE does not exist" >&2
fi