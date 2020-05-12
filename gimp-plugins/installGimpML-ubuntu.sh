if [ ! -d "gimpenv" ]; then

	echo "\n-----------Installing GIMP-ML-----------\n"
	sudo apt install python-minimal
	python2 -m pip install --user virtualenv
	python2 -m virtualenv gimpenv
	source gimpenv/bin/activate
	python2 -m pip install torchvision
	python2 -m pip install opencv-python
	python2 -m pip install numpy
	python2 -m pip install future
	python2 -m pip install torch
	python2 -m pip install scipy
	python2 -m pip install scikit-image
	python2 -m pip install typing
	python2 -m pip install enum
	python2 -m pip install pretrainedmodels
	deactivate

	echo "\n-----------Installed GIMP-ML------------\n"

else

	echo "Environment already setup!"

fi




