#ifndef KWS_RAW_PROBE_MANIFEST_H
#define KWS_RAW_PROBE_MANIFEST_H

/* Indices match model[] in train_c.c::buildModel (MODEL_SIZE == 17, per-conv
 * LayerNorm pre-ReLU). Each name identifies the tensor produced by that C layer's
 * forward, and is paired against the same-named PyTorch tensor in trace_pytorch.py. */
static const char *KWS_RAW_PROBES[17] = {
    "pool0",     /* 0  AvgPool1d ds */
    "conv1",     /* 1  Conv1d */
    "ln1",       /* 2  LayerNorm([16,1000]) */
    "relu1",     /* 3  ReLU */
    "pool1",     /* 4  MaxPool1d */
    "conv2",     /* 5  Conv1d */
    "ln2",       /* 6  LayerNorm([32,250]) */
    "relu2",     /* 7  ReLU */
    "pool2",     /* 8  MaxPool1d */
    "conv3",     /* 9  Conv1d */
    "ln3",       /* 10 LayerNorm([64,62]) */
    "relu3",     /* 11 ReLU */
    "pool3",     /* 12 MaxPool1d */
    "adaptpool", /* 13 AdaptiveAvgPool1d */
    "flatten",   /* 14 Flatten */
    "fc",        /* 15 Linear (logits) */
    "softmax",   /* 16 Softmax (probs) */
};

#endif
