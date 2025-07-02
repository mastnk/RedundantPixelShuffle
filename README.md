# Redundant Pixel Shuffle (Inverse and Restore)

This repository contains a pair of utility classes for spatial-to-channel and channel-to-spatial tensor transformations using overlapping patches. This technique can be used for data augmentation, local context encoding, or designing custom neural network layers inspired by inverse pixel shuffling.

---

## Overview

This module implements:

- **`InverseRedundantPixelShuffle`**  
  Extracts overlapping local patches (via sliding windows) from an image tensor and flattens them into the channel dimension.

- **`RedundantPixelShuffle`**  
  Restores the original image size from the expanded tensor, averaging overlapping regions to reconstruct the original values.

---

## Installation

Just copy the `*.py` file into your project.  
No extra dependencies other than `PyTorch`.

---

## Usage Example

```python
import torch
from redundant_pixel_shuffle import InverseRedundantPixelShuffle, RedundantPixelShuffle

# Example input
b, c, h, w = 2, 3, 3, 4
x = torch.arange(b * c * h * w, dtype=torch.float).view(b, c, h, w)

k = 3  # kernel size

# Transform
IRPS = InverseRedundantPixelShuffle(kernel_size=k)
RPS = RedundantPixelShuffle(kernel_size=k)

y = IRPS(x)
z = RPS(y)

print("Original x shape:", x.shape)
print("Transformed y shape:", y.shape)
print("Restored z shape:", z.shape)

