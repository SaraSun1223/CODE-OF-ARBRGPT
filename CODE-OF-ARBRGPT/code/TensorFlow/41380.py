import tensorflow as tf
import os

class DummyModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(1.0)

model = DummyModel()

ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
manager = tf.train.CheckpointManager(ckpt, directory="./checkpoints", max_to_keep=3)

# 第一次保存 —— 创建 checkpoint 目录和文件
manager.save()

# 模拟已有的 checkpoint 文件被占用或存在
with open("./checkpoints/checkpoint", "w") as f:
    f.write("lock this file")  # Windows 下此文件可能无法被 rename 替换

# 再次保存，触发 overwrite 情况
# 在 Windows 下，TensorFlow 会报错：Access is denied. rename failed
manager.save()
