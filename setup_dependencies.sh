#!/usr/bin/env bash

echo "Installing packages inside a virtual environment..."
conda env create -f environment.yml || echo "Installation failed!" && exit
conda activate torch_rgcn_venv
echo "Done. \n\n"


mkdir -p data
echo "Downloading AIFB..."
curl "https://www.dropbox.com/sh/ldjd70yvnu9akxi/AAAam7SBr5KXLfjk-NVGQNWRa?dl=1" -o "data/aifb.zip"
unzip "data/aifb.zip" -d "data/aifb"
echo "Done. \n\n"

echo "Downloading AM..."
curl "https://www.dropbox.com/sh/5ys1lfw9c8padz0/AABEJChkUHkxrWfvXrgehOX5a?dl=1" -o "data/am.zip"
unzip "data/am.zip" -d "data/am"
echo "Done. \n\n"

echo "Downloading BGS..."
curl "https://www.dropbox.com/sh/so1n0zc4zkel2mf/AACq3llckg1AAMfi2umI3MbGa?dl=1" -o "data/bgs.zip"
unzip "data/bgs.zip" -d "data/bgs"
echo "Done. \n\n"

echo "Downloading MUTAG..."
curl "https://www.dropbox.com/sh/tburaaxij0a1vmy/AAAlD5ORzcMbF3YpoynOLGqwa?dl=1" -o "data/mutag.zip"
unzip "data/mutag.zip" -d "data/mutag"
echo "Done. \n\n"

echo "Downloading FB15k..."
curl "https://www.dropbox.com/sh/rwcku99q10jzpzs/AACe4NgdH71AYV9hG7bxYMVTa?dl=1" -o "data/fb15k.zip"
unzip "data/fb15k.zip" -d "data/fb15k"
echo "Done. \n\n"

echo "Downloading FB15k-237..."
curl "https://www.dropbox.com/sh/b7c72uv9jmbwm7v/AABkasD__OutmGY0VZH7ZVBoa?dl=1" -o "data/fB15k-237.zip"
unzip "data/fB15k-237.zip" -d "data/fB15k-237"
echo "Done. \n\n"

echo "Downloading WN18..."
curl "https://www.dropbox.com/sh/egwgth011epusq7/AABWx1YWuEaMoumHDOknbCA9a?dl=1" -o "data/wn18.zip"
unzip "data/wn18.zip" -d "data/wn18"
echo "Done. \n\n"

echo "Downloading WN18RR..."
curl "https://www.dropbox.com/sh/e0c8axhazom8y9p/AAAP3CvZp9IZAophUkz6YZC3a?dl=1" -o "data/wn18rr.zip"
unzip "data/wn18rr.zip" -d "data/wn18rr"
echo "Done. \n\n"


echo "Setup complete."