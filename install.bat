:<<BATCH
    @echo off
    echo **** GIMP-ML Setup started ****
    python -m pip install virtualenv
	python -m virtualenv gimpenv3
	if "%1"=="gpu" (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html) else (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html)
    gimpenv3\Scripts\python.exe -m pip install GIMP-ML\.
    gimpenv3\Scripts\python.exe -c "import gimpml; gimpml.setup_python_weights()"
	echo **** GIMP-ML Setup Ended ****
    exit /b
BATCH
echo '**** GIMP-ML Setup started ****'
if python --version 2>&1 | grep -q '^Python 3\.'; then #
    echo 'Python 3 found.' #
else #
    if python3 --version 2>&1 | grep -q '^Python 3\.'; then #
        echo 'Python 3 found.' #
        alias python='python3' #
    fi #
fi #
python -m pip install virtualenv
python -m virtualenv gimpenv3 #
source gimpenv3/bin/activate #
python -m pip install GIMP-ML/.
python -c "import gimpml; gimpml.setup_python_weights()"
deactivate #
echo '*** GIMP-ML Setup Ended ****'
