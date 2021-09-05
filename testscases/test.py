import cv2
import gimpml

image = cv2.imread("sampleinput/img.png")
alpha = cv2.imread("sampleinput/alpha.png")

out = gimpml.edge(image)
cv2.imwrite("output/tmp-edge.jpg", out)

# out = gimpml.kmeans(image)
# cv2.imwrite("output/tmp-kmeans.jpg", out)
# #
# out = gimpml.deblur(image)
# cv2.imwrite('output/tmp-deblur.jpg', out)
#
# out = gimpml.deepcolor(image)
# cv2.imwrite('output/tmp-deepcolor.jpg', out)
#
# out = gimpml.dehaze(image)
# cv2.imwrite('output/tmp-dehaze.jpg', out)
#
# out = gimpml.denoise(image)
# cv2.imwrite('output/tmp-denoise.jpg', out)
#
out = gimpml.matting(image, alpha)
cv2.imwrite('output/tmp-matting.png', out)  # save as png

# out = gimpml.enlighten(image)
# cv2.imwrite('output/tmp-enlighten.jpg', out)

# face = cv2.imread('sampleinput/face.png')
# mask1 = cv2.imread('sampleinput/mask1.png')
# mask2 = cv2.imread('sampleinput/mask2.png')
# out = gimpml.newface(face[:, :, ::-1], mask1[:, :, ::-1], mask2[:, :, ::-1])
# cv2.imwrite('output/tmp-newface.png', out[:, :, ::-1])

# face = cv2.imread('sampleinput/face.png')
# out = gimpml.parseface(face[:, :, ::-1])
# cv2.imwrite('output/tmp-parseface.png', out[:, :, ::-1])

# mask1 = cv2.imread('sampleinput/mask1.png')
# mask2 = cv2.imread('sampleinput/mask2.png')
# out = gimpml.interpolateframe(mask1, mask2, 'output/interpolateframes')
#
img = cv2.imread("sampleinput/img.png")
out = gimpml.depth(img[:, :, ::-1])
cv2.imwrite(
    "output/tmp-depth.png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0]
)
#
# image = cv2.imread('sampleinput/face.png')
# out = gimpml.semseg(image[:, :, ::-1])
# cv2.imwrite('output/tmp-semseg.png', out[:, :, ::-1])
#
# image = cv2.imread('sampleinput/face.png')
# out = gimpml.super(image[:, :, ::-1])
# cv2.imwrite('output/tmp-super.png', out[:, :, ::-1])
#
# image = cv2.imread('sampleinput/inpaint.png')
# mask = cv2.imread('sampleinput/inpaint-mask.png')
# image = cv2.imread(r'C:\Users\Kritik Soman\GIMP-ML\cache1.png')
# mask = cv2.imread(r'C:\Users\Kritik Soman\GIMP-ML\cache0.png')
# out = gimpml.inpaint(image[:, :, ::-1], mask[:, :, 0])
# cv2.imwrite(r'C:\Users\Kritik Soman\GIMP-ML\res.png', out[:, :, ::-1])
