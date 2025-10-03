# RedundantPixelShuffle (Inverse & Restore)

This repository provides two utility classes that convert between spatial and channel representations using *overlapping patches*. You can use this for data augmentation, local context encoding, or as building blocks for custom neural network layers inspired by inverse pixel shuffle techniques.

## Overview

- **InverseRedundantPixelShuffle**  
  Extracts overlapping local patches from an image (via sliding windows) and flattens each patch into the channel dimension.

- **RedundantPixelShuffle**  
  Restores the original image from the expanded patch-channel representation. Overlapping regions are averaged to reconstruct the final image.

These transforms are complementary: one expands spatial information into channels, the other restores it.

## Installation

Just copy the `.py` files into your project.  
Requires only **PyTorch** (no additional dependencies).

## Usage Example

```python
import torch
from redundant_pixel_shuffle import InverseRedundantPixelShuffle, RedundantPixelShuffle

# Example input
b, c, h, w = 2, 3, 3, 4
x = torch.arange(b * c * h * w, dtype=torch.float).view(b, c, h, w)

k = 3  # kernel / patch size

irps = InverseRedundantPixelShuffle(kernel_size=k)
rps = RedundantPixelShuffle(kernel_size=k)

y = irps(x)
z = rps(y)

print("Original x shape:", x.shape)
print("Transformed y shape:", y.shape)
print("Restored z shape:", z.shape)
```

You should see that `z` approximates (or equals, depending on data) the original `x` in shape and content.

## Notes & Details

- **Overlap Averaging**  
  When reconstructing with `RedundantPixelShuffle`, overlapping pixels are averaged (or summed, depending on the implementation) to produce the final output.

- **Channel Ordering**  
  For `InverseRedundantPixelShuffle`, patches are flattened such that for each original channel (e.g. R, G, B), all positions in the kernel window are grouped together.

- **Edge Handling / Padding**  
  Care must be taken with boundaries; you may need to pad before applying inverse transform so that the reconstruction is consistent.

## Potential Use Cases

- Data augmentation by mixing spatial neighbors  
- Feature encoding that retains spatial locality  
- Custom layers for super-resolution, context modeling, or other vision tasks

## License & Contribution

Feel free to fork, modify, and use as you see fit. Contributions, bug reports, and pull requests are welcome!
