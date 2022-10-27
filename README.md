# Automated Immune Cell Segmentation

## Brief description of the repository

Code for recreating results in the publication <INSERT FINAL PAPER NAME>. Features classical image processing based techniques for single cell segmentation. Specifically this involves multiple thresholding methods for foreground/background differentiation, edge detection filters for border creation, and combination of methods through ensemble voting. `Overview_Image`

## Dependencies and system requirements (packages and versions)

Dependencies
* scipy (1.6.2)
* numpy (1.20.1)
* pandas (1.2.4)
* opencv-python (4.5.5.64)
* tifffile (2021.4.8)
* matplotlib (3.3.4)
* ipywidgets (7.6.3)
  
System Requirements
* 2.50GHz CPU
* 16.0 GB RAM
* 64-bit operating system
* Windows 10
* No GPU dependencies required

## How to install and run the code

1. Clone the GitHub Repository
2. Create an Anaconda environment.
3. Run through the `Immune_Cell_Segmentation_Notebook.ipynb`. If this is your first time running the notebook, ensure that you run the first cell for installing all files from `requirements.txt`
4. `base_path` is currently pointing to the demo data included above. to analyze your own data change the `base_path` to the location of your files. 
5. Continue running through cells to generate file lists and read in all requisite files for recreating results. 
6. Update parameters in `adpative segmentation` and proceed with data generation. Voila!  
7. Final spleen segmentation data was generated with the following parameters: 
    * Minimum ROI: 40
    * Lifetime Minimum: 800
    * Lifetime Maximum: 1500
8. Final tumor segmentation data was generated with the following parameters:
    * Minimum ROI: 60
    * Lifetime Minimum: 1000
    * Lifetime Maximum: 1500

## How to cite this code

```tex
https://github.com/skalalab/heaton_a-automated_immune_cell_segmentation.git@software{Peter_R_Rehani_2022_7213391},
  author       = {Peter R. Rehani and Alexa R. Heaton and Emmanuel Contreras Guzman and Melissa C. Skala},
  title        = {Statannotations},
  month        = dec,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.7213391},
  url          = {https://doi.org/10.5281/zenodo.7213391}
```
