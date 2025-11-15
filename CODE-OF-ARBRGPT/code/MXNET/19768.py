import gluoncv
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx
import numpy as np
from mxnet import init


name = "mask_rcnn_resnet18_v1b_coco"
mask_rcnn = gluoncv.model_zoo.get_model(name, pretrained=True, ctx=mx.cpu(0))
mask_rcnn.mask.deconv.bias.initialize(init.Constant(mx.nd.zeros(256)), force_reinit=True)
x, orig_img = data.transforms.presets.rcnn.load_test("biking-600.jpg")  # Replace biking-600.jpg with a real image path that you have
ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in mask_rcnn(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)
# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=mask_rcnn.classes, ax=ax)
plt.show()
print(np.sum(masks)) # This value stays the same whether you reinitialize the bias or not - which means it is not used
print(np.sum(scores))
print(np.sum(bboxes))