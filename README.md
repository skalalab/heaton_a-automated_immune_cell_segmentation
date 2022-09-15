# Automated Immune Cell Segmentation
## Brief description of the repository
Code for recreating results in the publication <INSERT FINAL PAPER NAME>. Features classical image processing based techniques for single cell segmentation. Specifically this involves multiple thresholding methods for foreground/background differentiation, edge detection filters for border creation, and combination of methods through ensemble voting.  INSERT COMPARISON IMAGE. 
## Dependencies and system requirements (packages and versions)
This will need to be pulled from Alexa's PC--the final conda environment will need to be generated. Instructions to do this can be found here. https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file 
Additionally, specify computer hardware based on your laptop's specifications as needed. The code for adaptive segmentation requires no GPU dependencies. <WILL NEED TO BE FILLED IN BASED ON ALEXA'S LAPTOP + CONDA ENVIRONMENT>  
<AFTER COMPLETION, PICK OUT SPECIFIC PACKAGE VERSIONS FOR REQUIREMENTS.TXT>  
## How to install and run the code
1. Clone the GitHub Repository
2. Create an Anaconda environment using the provided `environment.yml` file (generated above)
  - https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file specific instructions are presented here. 
3. Run through the `CD4_notebook.ipynb`. If this is your first time running the notebook, ensure that you run the first cell for installing all files from `requirements.txt`
4. Change `base_path` to the location of the stored data. 
5. Continue running through cells to generate file lists and read in all requisite files for recreating results. 
6. Update parameters in `adpative segmentation` and proceed with data generation. Voila!  
  - ALEXA WILL NEED TO SPECIFY THE EXACT COMBINATION OF PARAMETERS SHE USED FOR THE FINAL DATA GENERATION HERE

## How to cite this code (Alexa & Emmanuel will complete this section)
