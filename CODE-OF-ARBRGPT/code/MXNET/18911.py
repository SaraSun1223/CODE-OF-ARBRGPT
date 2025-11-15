
from mxnet import np, npx
from mxnet.gluon.data.vision import transforms

npx.set_np()

img=np.arange(11*9).reshape(11,9,1)

print(img[:, :, 0].P)

img_flip_lr = npx.image.random_flip_left_right(img)
print(img_flip_lr[:, :, 0])

img_flip_lr = npx.image.random_flip_left_right(img)  # try again...
print(img_flip_lr[:, :, 0])

transformer = transforms.Compose([
    transforms.RandomFlipLeftRight(),
    transforms.RandomFlipTopBottom()
])

print(transformer(img)[:, :, 0])
