# Exposure Fusion

Python implementation of Exposure Fusion.

Here are some photos with different exposures:

| ![A.jpg](./pics/A.jpg) | ![B.jpg](./pics/B.jpg) | ![C.jpg](./pics/C.jpg) | ![D.jpg](./pics/D.jpg) |
|------------------------|------------------------|------------------------|------------------------|

Here is the result of exposure fusion using the naive implementation:
![naive_fusion.jpg](./save/naive_fusion.jpg)

Here is the result of exposure fusion using the Gaussian kernel:
![gaussian_fusion.jpg](./save/gaussian_fusion.jpg)

Here is the result of exposure fusion using Gaussian pyramid and Laplacian pyramid:
![laplacian_fusion.jpg](./save/laplacian_fusion.jpg)

## Usage
1. Put photos with different exposures into the `pics` folder.
2. Run `main.py`

## References
[1]: T. Mertens, J. Kautz and F. Van Reeth, "Exposure Fusion," 15th Pacific Conference on Computer Graphics and Applications (PG'07), Maui, HI, USA, 2007, pp. 382-390, doi: 10.1109/PG.2007.17.
