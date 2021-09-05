:<<BATCH
    @echo off
    echo **** GIMP-ML Setup started ****
    python -m pip install virtualenv
	python -m virtualenv gimpenv3
	if "%1"=="gpu" (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html) else (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html)
    gimpenv3\Scripts\python.exe -m pip install GIMP-ML\.
    gimpenv3\Scripts\python.exe -c "import gimpml; gimpml.setup_python_weights()"
	echo **** GIMP-ML Setup Ended ****
    exit /b
BATCH
echo '**** GIMP-ML Setup started ****'
if python --version 2>&1 | grep -q '^Python 3\.'; then #
    echo 'Python 3 found.' #
    sudo apt-get install python3-distutils #
    sudo apt install python3-testresources #
    wget https://bootstrap.pypa.io/pip/get-pip.py #
    python GIMP-ML/get-pip.py #
    rm -rf get-pip* #
    python -m pip install virtualenv
    python -m virtualenv gimpenv3 #
    source gimpenv3/bin/activate #
    python -m pip install torch torchvision -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html #
    python -m pip install GIMP-ML/.
    python -c "import gimpml; gimpml.setup_python_weights()" #
    chmod -R 777 gimpenv3/lib/ #
    deactivate #
elif python3 --version 2>&1 | grep -q '^Python 3\.'; then #
    echo 'Python 3 found.' #
    sudo apt-get install python3-distutils #
    sudo apt install python3-testresources #
    wget https://bootstrap.pypa.io/pip/get-pip.py #
    python3 get-pip.py #
    rm -rf get-pip* #
    python3 -m pip install virtualenv
    python3 -m virtualenv gimpenv3 #
    source gimpenv3/bin/activate #
    python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html #
    python3 -m pip install GIMP-ML/.
    python3 -c "import gimpml; gimpml.setup_python_weights()" #
    chmod -R 777 gimpenv3/lib/ #
    deactivate #
else #
    echo 'Python 3 NOT found' #
fi #
echo '*** GIMP-ML Setup Ended ****'
