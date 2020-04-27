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
	pip install matplotlib
	pip install scipy
	pip install scikit-image
	pip install typing
	pip install albumentations
	pip install enum
	pip install pretrainedmodels
	cwd=$(pwd)
	echo -e "baseLoc='${cwd}/'\n$(cat colorize.py)" > colorize.py
	echo -e "baseLoc='${cwd}/'\n$(cat deblur.py)" > deblur.py
	echo -e "baseLoc='${cwd}/'\n$(cat deeplabv3.py)" > deeplabv3.py
	echo -e "baseLoc='${cwd}/'\n$(cat facegen.py)" > facegen.py
	echo -e "baseLoc='${cwd}/'\n$(cat faceparse.py)" > faceparse.py
	echo -e "baseLoc='${cwd}/'\n$(cat monodepth.py)" > monodepth.py
	echo -e "baseLoc='${cwd}/'\n$(cat super_resolution.py)" > super_resolution.py
	
	deactivate

	echo "\n-----------Installed GIMP-ML------------\n"

else

	echo "Environment already setup!"

fi


