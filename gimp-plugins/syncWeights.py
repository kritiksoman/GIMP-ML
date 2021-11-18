import os
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'
from gimpfu import *
activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))
import requests

def download_file_from_google_drive(id, destination,fileSize):
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()
	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)
	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)
	save_response_content(response, destination,fileSize)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination,fileSize):
	CHUNK_SIZE = 1 * 1024 * 1024
	n = 0
	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)
				n = n + 1
				gimp.progress_update(float(n)/float(fileSize))
				gimp.displays_flush() 

def syncGit(baseLoc):
	#git
	file_id = '17hdADAEWzHJCtSQrDyQs8v_MHTq1H04z'
	fileSize = 0.5 #in MB
	mFName = 'tmp.zip'
	destination = baseLoc + '/' + mFName
	gimp.progress_init("Downloading plugins (~" + str(fileSize) + "MB)...")
	download_file_from_google_drive(file_id, destination,fileSize)

	import zipfile
	with zipfile.ZipFile(destination, 'r') as zip_ref:
		zip_ref.extractall(baseLoc)

	import shutil
	root_src_dir = baseLoc + 'GIMP-ML-master/gimp-plugins/'
	root_dst_dir = baseLoc

	for src_dir, dirs, files in os.walk(root_src_dir):
		dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
		if not os.path.exists(dst_dir):
			os.makedirs(dst_dir)
		for file_ in files:
			src_file = os.path.join(src_dir, file_)
			dst_file = os.path.join(dst_dir, file_)
			if os.path.exists(dst_file):
				if os.path.samefile(src_file, dst_file):
					continue
				os.remove(dst_file)
			shutil.move(src_file, dst_dir)
	shutil.rmtree(baseLoc+'GIMP-ML-master')
	os.remove(baseLoc+'tmp.zip')
	
def sync(path,flag):
	if not os.path.isdir(path):
		os.mkdir(path)

	#deepmatting
	model = 'deepmatting'
	file_id = '11dxJKH8p7xkcGtMtvzMUw-ua6pZ0vrfw'
	fileSize = 108 #in MB
	mFName = 'stage1_sad_57.1.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#MiDaS
	model = 'MiDaS'
	file_id = '11eap5jc-4SCX_sMMxYE6Bi5q_BKw894a'
	fileSize = 143 #in MB
	mFName = 'model.pt'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#colorize
	model = 'colorize'
	file_id = '12tKfNIDewgJPbW3FiITV_AMbOtZWP0Eg'
	fileSize = 130 #in MB
	mFName = 'caffemodel.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#super_resolution
	model = 'super_resolution'
	file_id = '11GwnqKsYo2jujACD_GMB9uMTQfsuk2RY'
	fileSize = 6 #in MB
	mFName = 'model_srresnet.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#faceparse
	model = 'faceparse'
	file_id = '115nnWD0FoDkplTJYBY7lTQu1VNXFbCA_'
	fileSize = 51 #in MB
	mFName = '79999_iter.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#deblur
	model = 'deblur'
	file_id = '11Tt4a_URCer4ZxZA2l3dLMRVeSwoBFYP'
	fileSize = 233 #in MB
	mFName = 'mymodel.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)
	file_id = '11MCHMVhs4aaMGSusqiu0rtAo97xuC1GA'
	fileSize = 234 #in MB
	mFName = 'best_fpn.h5'
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#deeplabv3
	model = 'deeplabv3'
	file_id = '11rX1MHjhmtaoFTQ7ao4p6b31Oz300i0G'
	fileSize = 233 #in MB
	mFName = 'deeplabv3+model.pt'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#facegen
	model = 'facegen'
	ifolder = 'label2face_512p'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	if not os.path.isdir(path + '/' + model + '/' + ifolder):
		os.mkdir(path + '/' + model + '/' + ifolder)
	file_id = '122dREA3R0vsSWbrzBwhF5oqSEJ7yrRbL'
	fileSize = 342 #in MB
	mFName = 'latest_net_G.pth'
	
	destination = path + '/' + model + '/' + ifolder + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#deepdehaze
	model = 'deepdehaze'
	file_id = '1hrd310nYCbh6ui_ZsZci7Zna2AFP1sMS'
	fileSize = 0.008 #in MB
	mFName = 'dehazer.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

 	#deepdenoise
	model = 'deepdenoise'
	file_id = '1acZ1FTNMuAQaYtE3RYLA8fs8cQrW2tZ_'
	fileSize = 0.166 #in MB
	mFName = 'est_net.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)
	file_id = '1tBoyDxYJ92pvopBJeK9PmG_jMA_Ut38_'
	fileSize = 3 #in MB
	mFName = 'net.pth'
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#enlighten
	model = 'enlightening'
	file_id = '1V8ARc2tDgUUpc11xiT5Y9HFQgC6Ug2T6'
	fileSize = 35 #in MB
	mFName = '200_net_G_A.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

	#interpolateframes
	model = 'interpolateframes'
	file_id = '1bHmO9-_ENTYoN1-BNwSk3nLN9-NDUnRg'
	fileSize = 1.6 #in MB
	mFName = 'contextnet.pkl'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)
	file_id = '1cQvDPBKsz3TAi0Q5bJXsu6A-Z7lpk_cE'
	fileSize = 25.4 #in MB
	mFName = 'flownet.pkl'
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)	
	file_id = '1mlA8VtxIcvJfz51OsQMvWX24oqxZ429r'
	fileSize = 15 #in MB
	mFName = 'unet.pkl'
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

 	#inpainting
	model = 'inpainting'
	file_id = '1WmPevEnRcdUynVHL8pZNzHPmLQVCFjuE'
	fileSize = 132 #in MB
	mFName = 'model_places2.pth'
	if not os.path.isdir(path + '/' + model):
		os.mkdir(path + '/' + model)
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)
	file_id = '1hIcPqDp8JjzR5kmt275DaVgX2PEtahWS'
	fileSize = 148 #in MB
	mFName = 'refinement.pth'
	destination = path + '/' + model + '/' + mFName
	if not os.path.isfile(destination):
		gimp.progress_init("Downloading " + model +"(~" + str(fileSize) + "MB)...")
		download_file_from_google_drive(file_id, destination,fileSize)

