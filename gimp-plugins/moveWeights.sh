unzip weights.zip 'weights/*'
mkdir -p CelebAMask-HQ/MaskGAN_demo/checkpoints/label2face_512p
mv weights/facegen/* CelebAMask-HQ/MaskGAN_demo/checkpoints/label2face_512p/
rm -rf weights/
