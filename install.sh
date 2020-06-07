#!/bin/bash
set -e

if [ -d "gimpenv" ]; then
  echo "Environment already set up!"
  exit
fi

echo -e "\n-----------Installing GIMP-ML-----------\n"

if [ "$(uname)" == "Linux" ]; then
  if [[ $(lsb_release -rs) == "18.04" ]]; then #for ubuntu 18.04
    sudo apt-get install python3-minimal
  elif [[ $(lsb_release -rs) == "20.04" ]]; then #for ubuntu 20.04
    sudo apt-get install python3-minimal
    wget https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
  elif [[ $(lsb_release -rs) == "10" ]]; then #for debian 10
    sudo apt-get install gimp-python
    wget https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
  fi
elif [ "$(uname)" != "Darwin" ]; then
  echo "Warning: unsupported system '$(uname)'"
fi

python3 -m pip install --user --upgrade virtualenv
python3 -m virtualenv -p python3 gimpenv
source gimpenv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys; print(f'python3_executable = \'{sys.executable}\'')" > plugins/_config.py
deactivate

echo -e "\n-----------Installed GIMP-ML------------\n"
