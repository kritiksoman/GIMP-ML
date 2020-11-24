if [ ! -d "gimpenv" ]; then

	echo "\n-----------Installing GIMP-ML-----------\n"
	
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
	python -m pip install torchvision
	python -m pip install "opencv-python<=4.3"
	python -m pip install numpy
	python -m pip install future
	python -m pip install torch
	python -m pip install scipy
	python -m pip install typing
	python -m pip install enum
	python -m pip install pretrainedmodels
	python -m pip install requests
	deactivate

	echo "\n-----------Installed GIMP-ML------------\n"

else

	echo "Checking environment ..."

	source gimpenv/bin/activate
	WASOK=true

	pip list | grep -w torchvision
	if [ $? != 0 ]; then
		python -m pip install torchvision
		WASOK=false
	fi
	pip list | grep -w opencv-python
	if [ $? != 0 ]; then
		python -m pip install "opencv-python<=4.3"
		WASOK=false
	fi
	pip list | grep -w numpy
	if [ $? != 0 ]; then
		python -m pip install numpy
		WASOK=false
	fi
	pip list | grep -w future
	if [ $? != 0 ]; then
		python -m pip install future
		WASOK=false
	fi
	pip list | grep -w torch
	if [ $? != 0 ]; then
		python -m pip install torch
		WASOK=false
	fi
	pip list | grep -w scipy
	if [ $? != 0 ]; then
		python -m pip install scipy
		WASOK=false
	fi
	pip list | grep -w typing
	if [ $? != 0 ]; then
		python -m pip install typing
		WASOK=false
	fi
	pip list | grep -w enum
	if [ $? != 0 ]; then
		python -m pip install enum
		WASOK=false
	fi
	pip list | grep -w pretrainedmodels
	if [ $? != 0 ]; then
		python -m pip install pretrainedmodels
		WASOK=false
	fi
	pip list | grep -w requests
	if [ $? != 0 ]; then
		python -m pip install requests
		WASOK=false
	fi
	
	deactivate

	if [ "$WASOK" = true ]; then
		echo "Environment already setup!"
	else
		echo "Environment now correctly setup!"
	fi

fi


