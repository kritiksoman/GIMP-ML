#!/bin/bash

if [ -d "gimpenv" ]; then
	echo "Environment already set up!"
	exit
fi

echo -e "\n-----------Installing GIMP-ML-----------\n"

if [ "$(uname)" == "Darwin" ]; then
  # Running on Mac OS
  :
elif [ "$(uname)" == "Linux" ]; then
  # Assuming Ubuntu
  sudo apt install python-minimal gimp-python
  alias python='python2'
else
  echo "Unsupported system '$(uname)'"
  exit 1
fi

python -m pip install --user virtualenv
python -m virtualenv gimpenv
source gimpenv/bin/activate
python -m pip install -r requirements.txt
deactivate

echo -e "\n-----------Installed GIMP-ML------------\n"
