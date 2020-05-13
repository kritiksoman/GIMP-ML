unzip weights.zip
mkdir -p CelebAMask-HQ/MaskGAN_demo/checkpoints/label2face_512p
mkdir -p mkdir monodepth2/models/mono+stereo_640x192
mkdir -p pytorch-SRResNet/model
mkdir deeplabv3

mv weights/colorize/* neural-colorization/
mv weights/deblur/* DeblurGANv2/
mv weights/deeplabv3/* deeplabv3
mv weights/facegen/* CelebAMask-HQ/MaskGAN_demo/checkpoints/label2face_512p/
mv weights/faceparse/* face-parsing.PyTorch/
mv weights/monodepth/* monodepth2/models/mono+stereo_640x192/
mv weights/super_resolution/* pytorch-SRResNet/model/
rm -rf weights/
