# AT-TPC Project  
Collaborators (order is not relevant): 
Andreas Ceulemans(AC), Stefano Fracassetti(SF), Ahmed Youssef(AY), Haoran Sun(HS) and Luca Morselli(LM)

# Table of contents:
## 1) Recent updates on the work: 
This session provides the latest information on the work done. If you do something on your branch worth sharing, use this session! Conversely, if you need to consult what your colleagues have recentrly done, look at this session!  

## 2) Work Organization:  
This session describes how we manage the collaboration, through this GitHub folder.  

## 3) Important information on the project:  
This session describes what this project is meant to look like, and our final goal. 

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

## 1) Recent updates on the work: (please indicate the date and your name) 

### 2020_08_25 at 16:00 CET : Meeting with the collaboators  
We planned a skype meeting in order to discuss on the project, join all the material, and finilized the report.  

### 2020_08_25 SF: Completed the AutoEncoder + KMeans (by Luca Morselli)  
I have coped Luca file on autoencoder and modified in my branch. It is now working and finilised. I have used some functions in helper_functions, in order to reduce the code. I have also used different number of cluster with kmeans ans choose the best model (using "best_km_custer" function).  

### 2020_08_24 AC: Updated report + helper functions.
I have added a new report version adding some sections from my projectfile to my branch. Minor edit helper function.
See AT_Report.ipynb and helper_functions.py in my branch


### 2020_08_24 SF: Finalizing CNN (together with Andreas)   
I have worked on the CNN model, now it perfroms well. I have used two model , one costumized and the VGG16.   


### 2020_08_21 SF: Updating best_cl_km (helper_function.py).  
Now this function accepts all possible number of clusters (not only 3 as before), and find the correct match cluster=labels.  
The function is in helper_function.py of my branch.   


### 2020_08_19 SF: Trying CNN (using VGG16).  
I have upated my AT_Project file: I have converted (x,y) projection in images, and then I tried to apply CNN. At this point it seems not to work. I will debug it in the next days, and try again. Have a look at this file in case you want to try CNN as well. It may be useful. 


### 2020_08_18 SF: Labels changed. 
The correct labels are even=beam, odd=reaction. I am working on that.  


### 2020_08_18 SF: Most updated file versions.  
- You can find the most updated code in AT_Report in my branch (SFprojectfile). This Report contains the description of all the phase that can be called "Data import and Data Visualization".  It contains also the description (and implementation) of some models: Logistic Regression, Random Forest, DNN, and K-Means. The code of these models has been taken from AC branch, and I have added some description, and optimize the output for better visualization. At this point, the comments/description of the various model can be considered only a draft. More info can be added, better grammar can be used.  
- Also the most updated version of the helper_function.py it is in the same branch. Please use these file as starting point, to avoid extra work.  
-The most updated version of the Project (on Logistic Regression, Random Forest, DNN, and K-Means)is the one of Andreas, but of course everyone is working on different Models, so this is not so important.

## 2) Work Organization: 
BRANCHES:  
Everyone is taking a different branch. Anyway, more people can work simultaneously on the same file (or same portion of code). Conflicts must be resolved manually carefully. Please, if your are not sure what to do, ask before commits.  
Rule: you can always commit to your branch (it is your personal space), but if you want to merge ask permission, or discuss with your colleagues.  
Maybe some of this is obvious to some of you, but not everyone has use Git extensively before.



## 3) Important information on the project  
AT-TPC Final Project - TALENT School on Machine learning  
Deadline: 31/08/2020  

What is expected from us?  
Out goal is to deliver a report in which we perform the desired task, as indicated in the project assgined.
The report (computational project = jupyter notebook) would go through every step of out analysis and describe what we did and how,  
possible it would be nice to add extra information, or brief description on the model used. It would be a nice way in order to review this concept, and to explain your colleagues what you did, if they are not very familiar with it.  

AT_Report: This file will be the final report. In the end, it should be run independently, and be more accurate as possible in describing out work.
AT_Project: This file is the "code" where we try things. In the end it should look very similar to the report, be with only really needed description or comments. In principle, this file should not be delivered, since every infomration should be already included and described in the Report.  

The idea is that one perform taks in the Project file, he/she tests it, and only when it works just fine, he/she import the code in the report, where he/she adds extra information, and produce better outputs. 
