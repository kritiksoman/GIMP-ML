if [ ! -d "gimpenv" ]; then

	echo "\n-----------Installing GIMP-ML-----------\n"

	python -m pip install --user virtualenv
	python -m virtualenv gimpenv
	source gimpenv/bin/activate
	pip install torchvision
	pip install opencv-python
	pip install numpy
	pip install future
	pip install torch
	pip install scipy
	pip install scikit-image
	pip install typing
	pip install enum
	pip install pretrainedmodels

	
	deactivate

	echo "\n-----------Installed GIMP-ML------------\n"

else

	echo "Environment already setup!"

fi


