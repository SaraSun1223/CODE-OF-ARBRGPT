import mxnet as mx

from mxnet.gluon import nn

import os

os.environ['MXNET_EXEC_INPLACE_GRAD_SUM_CAP'] = '4'

os.environ['DMLC_LOG_STACK_TRACE_DEPTH'] = '20'



mx.npx.set_np()



ctx = mx.gpu()



batch_size = 2

sequence_length = 10



mask = mx.np.random.randint(0, 2, (batch_size, sequence_length), ctx=ctx)

contextual_embeddings = mx.np.random.normal(0, 1, (2, sequence_length, 256), ctx=ctx, dtype=mx.np.float32)



p_mask = 1 - mask



l_start_scores = nn.Dense(1, flatten=False)

l_end_scores = nn.Dense(1, flatten=False)

l_start_scores.initialize(ctx=ctx)

l_end_scores.initialize(ctx=ctx)

with mx.autograd.record():

    start_scores = mx.np.squeeze(l_start_scores(contextual_embeddings), -1)

    start_logits = start_scores * p_mask + (1 - p_mask) * (-1e18)

    contextual_embeddings = mx.np.expand_dims(contextual_embeddings, axis=1)  # (B, 1, T, C)

    end_scores = l_end_scores(contextual_embeddings)

    end_scores = mx.np.squeeze(end_scores, -1)

    p_mask = mx.np.expand_dims(p_mask, axis=-1)

    end_logits = p_mask * end_scores + (1 - p_mask) * -1e18

    end_logits = end_logits * p_mask + (1 - p_mask) * -1e18

    loss = end_logits.sum()

loss.backward()

mx.npx.waitall()