#!/bin/bash

if [ -d "gimpenv" ]; then
  echo "Environment already set up!"
  exit
fi

echo -e "\n-----------Installing GIMP-ML-----------\n"

if [ "$(uname)" == "Linux" ]; then
  if [[ $(lsb_release -rs) == "18.04" ]]; then #for ubuntu 18.04
    sudo apt-get install python-minimal
    alias python='python2'
  elif [[ $(lsb_release -rs) == "20.04" ]]; then #for ubuntu 20.04
    sudo apt-get install python2-minimal
    wget https://bootstrap.pypa.io/get-pip.py
    alias python='python2'
    python get-pip.py
  elif [[ $(lsb_release -rs) == "10" ]]; then #for debian 10
    sudo apt-get install gimp-python
    wget https://bootstrap.pypa.io/get-pip.py
    python get-pip.py
  fi
elif [ "$(uname)" != "Darwin" ]; then
  echo "Warning: unsupported system '$(uname)'"
fi

python -m pip install --user virtualenv
python -m virtualenv gimpenv
source gimpenv/bin/activate
python -m pip install -r requirements.txt
deactivate

echo -e "\n-----------Installed GIMP-ML------------\n"
