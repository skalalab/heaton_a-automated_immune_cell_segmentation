[![DOI](https://zenodo.org/badge/479512170.svg)](https://zenodo.org/badge/latestdoi/479512170)

# Automated Immune Cell Segmentation

## Brief description of the repository

Code for recreating results in the publication: *Single cell metabolic imaging of tumor and immune cells in vivo in melanoma bearing mice*. Features classical image processing based techniques for single cell segmentation. Specifically, this involves multiple thresholding methods for foreground/background differentiation, edge detection filters for border creation, and combination of methods through ensemble voting. 
![Overview Image](Overview_Image.png)

## Dependencies and system specifications (packages and versions)

Dependencies
* scipy (1.6.2)
* numpy (1.20.1)
* pandas (1.2.4)
* opencv-python (4.5.5.64)
* tifffile (2021.4.8)
* matplotlib (3.3.4)
* ipywidgets (7.6.3)
  
System Specifications
* 2.50GHz CPU
* 16.0 GB RAM
* 64-bit operating system
* Windows 10
* No GPU dependencies required

## How to install and run the code

1. Clone the GitHub Repository
2. [Create an Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
3. Run through the `Immune_Cell_Segmentation_Notebook.ipynb`. If this is your first time running the notebook, ensure that you run the first cell for installing all files from `requirements.txt`
4. `base_path` is currently pointing to the demo data included above. To analyze your own data change the `base_path` to the location of your files. 
5. Continue running through cells to generate file lists and read in all requisite files for recreating results. 
6. Update parameters in `adpative segmentation` and proceed with data generation. Voila!  
7. Final spleen segmentation data was generated with the following parameters: 
    * Save Masks: True
    * Minimum ROI: 40
    * Include Lifetime: True
    * Lifetime Minimum: 800
    * Lifetime Maximum: 1500
8. Final tumor segmentation data was generated with the following parameters:
    * Save Masks: True
    * Minimum ROI: 60
    * Include Lifetime: True
    * Lifetime Minimum: 1000
    * Lifetime Maximum: 1500

## How to cite this code

```tex
  author       = {Alexa R. Heaton, Peter R. Rehani, Anna Hoefges, Angelica F. Lopez, Amy K. Erbe, Paul M. Sondel, and Melissa C. Skala},
  title        = {Single cell metabolic imaging of tumor and immune cells in vivo in melanoma bearing mice},
  month        = march,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.7696762},
  url          = {https://github.com/skalalab/heaton_a-automated_immune_cell_segmentation}
```
