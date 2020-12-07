if [ ! -d "gimpenv" ]; then

	echo "-----------Installing GIMP-ML-----------"
	
	if [ "$(uname)" == "Linux" ]; then
		if [[ $(lsb_release -rs) == "18.04" ]]; then #for ubuntu 18.04
			sudo apt install python-minimal
			alias python='python2'
		       
		elif [[ $(lsb_release -rs) == "20.04" ]]; then #for ubuntu 20.04
			sudo apt install python2-minimal
			wget https://bootstrap.pypa.io/get-pip.py 
			alias python='python2'
			python get-pip.py	
			sudo apt-get install libpython2.7

		elif [[ $(lsb_release -rs) == "10" ]]; then #for debian 10
			wget https://bootstrap.pypa.io/get-pip.py 
			python get-pip.py
		fi
	fi

	python -m pip install --user virtualenv
	python -m virtualenv gimpenv
	source gimpenv/bin/activate
	python -m pip install -r requirements.txt
	deactivate

	echo "-----------Installed GIMP-ML------------"

else

	echo "------Checking GIMP-ML environment------"


	source gimpenv/bin/activate
	python -m pip install -r requirements.txt
	deactivate

	echo "------------GIMP-ML updated-------------"

fi


