# Deep Learning On Medical 3D Imaging

## Medical 3D Imaging overview
For a better comprehension of Medical 3D Imaging ecosystem, we need to get a handle of following aspects:

- Medical 3D Imaging DEVICES
   - Medical 3D Imaging (hardware players) VENDORS 
   - Medical 3D Imaging SOFTWARES
- Medical Imaging Stoarge & Exchnage protocol / Standard (DICOM)
- Medical 3D Imaging DATASET formats
- Medical 3D Imaging DATASET Handling/ PRE-PROCESSING PYTHON Libraries

Mediacal Images comes from various Medical 3D Imaging Devices; & are generally, stored, exchanged / transmissted as **DICOM** images.


## Medical 3D Imaging DEVICES

 -   MRI Scan (magnetic resonance imaging)
 -   CT Scan (computed tomography)
 -   Xray Scans
 -   PET Scan (positron emission topography)
 -   PET/CT or PET/MRI Scans
 -   InfraRed Images
 -   HD Video Camera Recordeers for Medical Pre/Intra/Post 'Operative Procedures'

#### Medical 3D Imaging (hardware players) VENDORS
- Philips Healthcare
- Siemens Healthineers AG
- GE Healthcare Inc.
- Hitachi Medical Corporation
- Carestream Health Inc.


#### Medical 3D Imaging SOFTWARES
- 3D Rendering Software
- 3D Scanning Software
- 3D Modelling Software


<p align="center">**3D Medical Imaging (Vantage View)**</p>
<p aling="center">
<img alt="medical imaging" src="https://www.researchgate.net/profile/Nilanjan_Dey3/publication/312222298/figure/fig2/AS:476035763445761@1490507178297/Association-between-the-electromagnetic-spectrum-and-the-medical-modalities.png"></p>




## Medical Imaging Stoarge & Exchnage protocol | DICOM (So, What & Why about DICOM?)

DICOM Images:**Digital Imaging and Communications in Medicine**: The standard for (& most commonly used for)
 - **communication and management** of medical imaging information and related data. 
 - **storing and transmitting** medical images enabling the *integration of medical imaging devices such as scanners, servers, workstations, printers, network hardware, and **picture archiving and communication systems (PACS)** from multiple manufacturers.*
 - widely adopted by **hospitals** and is making inroads into smaller applications like **dentists' and doctors' offices**.

* File Format — All patient medical images are saved in the **DICOM file format**. This format has **PHI (protected health information)** about the patient such as — name, sex, age in addition to other image related data such as equipment used to capture the image and some context to the medical treatment. Medical Imaging Equipments create DICOM files. 

* DICOM Viewer: Doctors use DICOM Viewers, computer software applications that can display DICOM images, read and to diagnose the findings in the images. 
A useful DICOM viewer for Windows users - [Mango](https://idoimaging.com/programs/124)


* Communications Protocol — The **DICOM communication protocol** is used to search for imaging studies in the archive and restore imaging studies to the workstation in order to display it. All medical imaging applications that are connected to the hospital network use the DICOM protocol to exchange information, mainly DICOM images but also patient and procedure information. There are also more advanced network commands that are used to control and follow the treatment, schedule procedures, report statuses and share the workload between doctors and imaging devices.

## Medical 3D Imaging DATASET formats 
Image file format is often a confusing aspect for someone wishing to process medical images. 

A medical image data set consists typically of 
 - one or more images representing the projection of an anatomical volume onto an image plane (projection or planar imaging), 
 - a series of images representing thin slices through a volume (tomographic or multislice two-dimensional imaging), 
 - a set of data from 
      - a volume (volume or three-dimensional imaging), or 
      - multiple acquisition of the same tomographic or 
      -  volume image over time to produce a dynamic series of acquisitions (four-dimensional imaging). 
       
The file format describes how the image data are organized inside the image file and how the pixel data should be interpreted by a software for the correct loading and visualization.

* Data file extensions for MRI/CT/X-Ray scan 
  1. <strong>.dcm - DICOM </strong>
  2. .nii (Neuroimaging Informatics Technology Initiative (**Nifti**))
  3. .mnc (**Minc**)
  4. .img and .hdr (**Analyze**)
  
P.S.:
|#|Format | Header | Extension |
| :---: | :---: | :---: | :---: |
|1| DICOM | 	Variable length binary format| .dcm|
|2|Nifti | Fixed-length: 352 byte binary format| .nii|
|3|Minc| Extensible binary format| .mnc|
|4| Analyze| Fixed-length: 348 byte binary format| .img and .hdr|

Reference:
[Data file extensions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3948928/)

## Medical 3D Imaging DATASET PRE-PROCESSING/ Handling PYTHON Libraries
 Python library that handle MRI/CT Scan/X-Ray imaging data
   * [pydicom](https://pydicom.github.io/) - A very popular python package used for analyzing DICOM images.
   * [MedPy](https://pypi.org/project/MedPy/) - Another populat python library 

Python libraries to handle Infrared images
   * [Pyradi](https://pypi.org/project/pyradi/)
   * [Spy](http://www.spectralpython.net/)
   * [Rampy](https://github.com/charlesll/rampy)

[White paper for reference.](https://sci-hub.tw/https://ieeexplore.ieee.org/document/7532521)   
   ```Python
   pip install pydicom
   pip install pydicom medpy nibabel
   pip install pyradi spectral rampy
   ```
   
#### Medical Imaging manipulation & vectorization
    
   * [How to manipulate and vectorize data](https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial)
   * Video tutorial on such data can be found [here](https://www.youtube.com/watch?v=KlffppN47lc)
   * [Simple vectorization process for dicom images](https://github.com/Deep-Mind-Hive/2D-imaging-DL/blob/master/vectorizing.py)
   
   

### MRI/CT-Scan/X-rays/ Infrared Scanning - An Overview
#### MRI Background

Magnetic Resonance Imaging (MRI) is the most common diagnostic tool brain tumors due primarily to it's noninvasive nature and ability to image diverse tissue types and physiological processes. MRI uses a magnetic gradient and radio frequency pulses to take repetitive axial slices of the brain and construct a 3-dimensional representation. Each brain scan 155 slices, with each pixel representing a 1mm<sup>3</sup> voxel.  
<p align="center">
<img alt="Basic MRI Workflow" src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/MRI_workflow.png" width=450>
<img alt="3D rendering produced by T2 MRI scan" src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/t29_143.gif" width=250>  </p>


#### CT-Scan Background
A computerized tomography scan (CT or CAT scan) uses computers and rotating X-ray machines to create cross-sectional images of the body. These images provide more detailed information than normal X-ray images. They can show the soft tissues, blood vessels, and bones in various parts of the body. [(Read more...)](https://www.healthline.com/health/ct-scan)
<p align="center">
<img alt="Basic MRI Workflow" src="https://caraccidentsinorlando.com/wp-content/uploads/2017/12/ct-vs-mri.jpg" width=450></p>


#### MRI/CT-Scan/X-rays pre-processing ([code](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/brain_pipeline.py))

One of the challenges in working with MRI/CT-Scan/X-rays data is dealing with the artifacts produced either by inhomogeneity in the magnetic field or small movements made by the patient during scan time. Oftentimes a bias will be present across the resulting scans, which can effect the segmentation results particularly in the setting of computer-based models.

<p align="center">
<img alt="Bias correction before and after" src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/n4_correction.png" width=200>  </p>


#### Infrared Background
In infrared photography, the film or image sensor used is sensitive to infrared light. The part of the spectrum used is referred to as near-infrared to distinguish it from far-infrared, which is the domain of thermal imaging. Wavelengths used for photography range from about 700 nm to about 900 nm. Film is usually sensitive to visible light too, so an infrared-passing filter is used; this lets infrared (IR) light pass through to the camera, but blocks all or most of the visible light spectrum

<p align="center">
<img alt="InfraRed image" src="http://www.thermomed.org/images/rost-regulation_470.jpg" ></p>



## Medical Imaging 'Annotated Dataset' Sources:

One of the prominent Data Source with proper labeling - 
[Data Science Bowl](https://datasciencebowl.com/)
 -  [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data)
 -  [data-science-bowl-2017](https://www.kaggle.com/c/data-science-bowl-2017/data)
 
 <p align="center">
<img alt="segments of dataset" src="https://github.com/naldeborgh7575/brain_segmentation/raw/master/images/segment.png"></p>

## CNN models work for Image featurization

![picture alt](https://www.researchgate.net/publication/314282902/figure/fig1/AS:469481303613440@1488944473917/Architecture-of-the-proposed-CNN-model-with-2-convolutional-layers.png "Title is optional")



Convolutional Neural Network is a neural network consists of convolutional layer, pooling layer (max pooling or average pooling but max pooling is preferable), fully connected layer and at last a softmax pooling

