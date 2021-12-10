#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1K-pgnHm6cWfV7q9Enhdgu1yDNGEKG3cH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K-pgnHm6cWfV7q9Enhdgu1yDNGEKG3cH" -O tmp.zip && rm -rf /tmp/cookies.txt
unzip tmp.zip -d "./data/"
rm tmp.zip