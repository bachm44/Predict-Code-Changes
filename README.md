# Early Prediction for Merged vs Abandoned CodeChanges in Modern Code Reviews

This is a tool to predict whether a code change request will be merged or abandoned as
soon as it has been uploaded and can update its predictions with each new revision. The tool is developed for Gerrit based code review systems. Its objective is to help
the code reviewers prioritize change requests based on the probability of getting merged.
Hence, saving their time and efforts. <b> We call this tool PredCR.  PredCR is a LightGBM
based machine learning classifier which can,</b>

* Predict whether a code change request will be merged with on average 85% AUC score as soon as the
change request is submitted and assigned to a reviewer. Improving the state-of-the-art [[1]](#1)
  by 18-28%.
* Even for new authors, predict merge probability with on average 78% AUC score. Improving the state-of-the-art [[1]](#1)
  by 21-31%.
* Provides two adjusted approaches to update prediction for each new revision of the same change request,
  and maintains significant results.
  
* Complete training within several seconds, hence feasible to use in real world projects.
  
We have mined changes from the following Gerrit projects

* [Eclipse](https://git.eclipse.org/r)
* [Gerrithub](https://review.gerrithub.io)
* [Libreoffice](https://gerrit.libreoffice.org)

## Project Structure

The root folder contains the [`Config`](Config.py) file.

* <b> Config</b> : Basic configuration for data path and models. Reduce the number of multiple runs here if
    you want the results fast. Change which project to run models on.

There are three subdirectories of the root folder.

### [1. Data](Data)

List of subdirectories:

* Eclipse
* Gerrithub
* Libreoffice

Each project directory contains their features and experimentation
  results. After mining the raw data, they will be stored here also. For
  file size limit in GitHub we are unable to upload them here. But they are
shared in this [Google Drive](https://drive.google.com/drive/folders/1z2KmxgYNgO5sNBHZLb_Nm43bFqH4vi2g?usp=sharing).
  You can download the raw dataset for a single project too from there and calculate the features from scratch if
  you want. Just unzip the files and keep the folder structure same as shown in Section [Mining](#mining).

### [2. Results](Results)

Contains csv result files for each project in their respective subdirectory. Each folder has feature importance,
train and test results for each fold, overall results for our and Fan et al.'s [[1]](#1) model.

### [3. Source](Source)

Contains source codes necessary for the project. It has three subdirectories :

* Experiments
* Feature Calculator
* Miners

It also has the [`Util.py`](Source/Util.py) file. Which contains some util methods used by other files.

#### [3.1 Experiments](Source/Experiments)

Source codes for running all the experiments mentioned in the paper.

* <b>Calculate developer effort</b>: Calculates developer effort in terms of duration of days, number of messages and number of changes per code change.
* <b>Complete mining process</b>: Contains complete raw change data mining steps
    (except file diff).

* <b>Cross project validation</b>: Calculates model performance across projects.
* <b>DNN model</b>: Contains the DNN model we built to find the best classifier for change prediction.

#### [3.2 Feature Calculators](Source/Feature%20Calculators)

Contains feature calculation related files.

* <b>Feature calculator</b>: Calculates feature sets from raw data created after
  mining.
* <b>Feature calculator for Fan</b>: Calculates feature for state-of-the-art work by
  Fan et al.[[1]](#1). Their shared repository can be found [here](https://github.com/YuanruiZJU/EarlyPredictionReview).

* <b>Feature calculator for multiple revisions</b>: Calculates features when prediction
  is updated for each new revision of the change request.

* <b>Longitudinal 10-fold cross validation</b>: Runs each of the experiments presented is our
  paper with longitudinal cross validation setup.

* <b>Longitudinal 10-fold cross validation - Fan</b>: Runs each of the experiments for Fan's [[1]](#1)
  work we have compared in our paper, with longitudinal cross validation setup.

#### [3.3 Miners](Source/Miners)

Contains the files necessary to mine the raw code changes and related data from Gerrit projects.

* <b>Mine file diff </b>: Mines file diff data for first revision of each selected code changes.
  Used later to calculated code segment related features.

* <b>Miner</b>: Contains the miner class implementation, used to mine code changes from Gerrit.
* <b> SimpleParser </b>: Parses the json responses from Gerrit server and return them in Class.
  
## How to run

Open `Predict-Code-Change` as project in Pycharm or any other python IDE. People interested in just testing the
tool should directly jump to [Experimentation](#exp) section. All the data need for running the experiments are
already uploaded. If you want to run them on your own mined dataset, complete the following two sections first.

### <a id="mining">Mining</a>

* Run the [`Source/Miners/Complete Mining Process.py`](Source/Miners/Complete%20mining%20process.py) file to start mining.
* Set the project name, make sure Gerrit class has corresponding Gerrit server address for this project.
* Check if the directories for the data to be dumbed is created.
* Set the start and end time period, changes created and closed in that period will be collected. Make
  sure the time format is valid. Check existing code in [`Source/Miners/Miner.py`](Source/Miners/Miner.py) for example.
* This step is long and time-consuming. Specially, because during downloading large chunks of
data, Gerrit servers randomly close the connection, and you have to rerun the miner several times
till it is successful in mining all changes within a period.

* <b> For best experience, run each steps in the miner individually. </b> When you want to run one, comment out the others.
Gerrit change response collected by [`Source/Miners/Miner.py`](Source/Miners/Miner.py) doesn't contain file contents.
* Run the [`Source/Miners/Mine file diff.py`](Source/Miners/Mine%20file%20diff.py) after completing previous steps, to mine
file contents for first revision of each selected change request.
* This miner doesn't batch download, so expect a looooong time to finish
mining changes. Also, occasionally Gerrit will close connections or send
  response not found messages. Rerunning the miner might fix that issue sometimes.
  
* With mined data project structure will look similar to this for each project
  * Eclipse
    * change : Batch of change requests
    * changes: Individual change requests.
    * diff: File diff content for first revision of each change request.
    * profile: Profile of Gerrit authors.

### <a id="feature_calculation">Feature calculation</a>

* This step can run only after completing previous mining steps.
* Currently, our raw data isn't added here.
* Run [`Source/Feature Calculators/Feature calculator.py`](Source/Feature%20Calculators/Feature%20calculator.py) to
 calculate features from raw data. It sorts selected changes from `Project_selected_change_list.csv` and searches
 for their corresponding mined files to calculate feature.
* So for each change in the selected list, the followings must be present before
  calculating feature. An example is present in Eclipse project.
  * Project
    * changes
      * Project_changeNumber_change.json
    * diff
      * Project_changeNumber_diff.json
    * profile
      * profile_accountId.json
  
### <a id="exp">Experimentation</a>

* Set config values in file [`Config.py`](Config.py). For example project, number of runs, folds,
  feature list, data path, seed.
  
* Running [`Source/Experiments/Longitudinal 10 fold cross validation.py`](Source/Experiments/Longitudinal%2010%20fold%20cross%20validation.py) with run our model experiments for `project`
  as specified in [`Config.py`](Config.py) file, `runs` times, using `folds` number of folds. It will generate
  the following files in `Data\project` folder :
  * <b>project_train_result_cross.csv </b>: Average train performance for each fold.
    *<b> project_test_result_cross.csv </b>:Average test performance for each fold.
  * <b>project_result_cross.csv</b>: Overall average performance across folds.
  * <b>project_feature_importance_cross.csv</b>: Average feature importance calculated from LightGBM
    `feature_importances_` attribute.
  
* Similarly, running [`Source/Experiments/Longitudinal 10 fold cross validation - Fan.py`](Source/Experiments/Longitudinal%2010%20fold%20cross%20validation.py) with run Fan's [[1]](#1)
  model experiments for parameters specified in [`Config.py`](Config.py) file.

## Citation

Paper link : [IST](https://www.sciencedirect.com/science/article/abs/pii/S0950584921002032), [arxiv](https://arxiv.org/pdf/1912.03437.pdf).

```bash
@article{ISLAM2022106756,
title = {Early prediction for merged vs abandoned code changes in modern code reviews},
journal = {Information and Software Technology},
volume = {142},
pages = {106756},
year = {2022},
issn = {0950-5849},
doi = {https://doi.org/10.1016/j.infsof.2021.106756},
url = {https://www.sciencedirect.com/science/article/pii/S0950584921002032},
author = {Khairul Islam and Toufique Ahmed and Rifat Shahriyar and Anindya Iqbal and Gias Uddin}
}
```

## References

<a id="1">[1]</a>
[Y. Fan, X. Xia, D. Lo, S. Li, Early prediction of merged code changes to prioritizereviewing tasks,
Empirical Software Engineering (2018) 1–48.](https://link.springer.com/content/pdf/10.1007/s10664-018-9602-0.pdf)
