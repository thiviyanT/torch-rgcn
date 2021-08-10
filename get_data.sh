#!/bin/bash

mkdir -p data
echo "Downloading AIFB..."
wget "https://www.dropbox.com/sh/ldjd70yvnu9akxi/AAAam7SBr5KXLfjk-NVGQNWRa?dl=1" -O "./data/aifb.zip"
unzip "data/aifb.zip" -d "data/aifb"
echo -e "Done. \n\n"

echo "Downloading AM..."
wget "https://www.dropbox.com/sh/5ys1lfw9c8padz0/AABEJChkUHkxrWfvXrgehOX5a?dl=1" -O "data/am.zip"
unzip "data/am.zip" -d "data/am"
echo -e "Done. \n\n"

echo "Downloading BGS..."
wget "https://www.dropbox.com/sh/so1n0zc4zkel2mf/AACq3llckg1AAMfi2umI3MbGa?dl=1" -O "data/bgs.zip"
unzip "data/bgs.zip" -d "data/bgs"
echo -e "Done. \n\n"

echo "Downloading MUTAG..."
wget "https://www.dropbox.com/sh/tburaaxij0a1vmy/AAAlD5ORzcMbF3YpoynOLGqwa?dl=1" -O "data/mutag.zip"
unzip "data/mutag.zip" -d "data/mutag"
echo -e "Done. \n\n"

echo "Downloading FB-Toy..."
wget "https://www.dropbox.com/sh/5kv7xk4cj1md9zw/AADpaREEK9K5NX_Vb5eRcXuRa?dl=1" -O "data/fb-toy.zip"
unzip "data/fb-toy.zip" -d "data/fb-toy"
echo -e "Done. \n\n"

echo "Downloading WN18..."
wget "https://www.dropbox.com/sh/egwgth011epusq7/AABWx1YWuEaMoumHDOknbCA9a?dl=1" -O "data/wn18.zip"
unzip "data/wn18.zip" -d "data/wn18"
echo -e "Done. \n\n"
