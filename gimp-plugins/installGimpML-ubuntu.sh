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
	python2 -m pip install matplotlib
	python2 -m pip install scipy
	python2 -m pip install scikit-image
	python2 -m pip install typing
	python2 -m pip install albumentations
	python2 -m pip install enum
	python2 -m pip install pretrainedmodels
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




