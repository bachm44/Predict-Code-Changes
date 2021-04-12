# Early Prediction for Merged vs Abandoned CodeChanges in Code Reviews
This is a tool to predict whether a code change request will be merged or abandoned as
soon as it has been uploaded and can update its predictions with each new revision. 
The tool is developed for Gerrit based code review systems. Its objective is to help
the code reviewers prioritize change requests based on the probability of getting merged.
Hence, saving their time and efforts. <b>We call this tool PredCR.  PredCR is a LightGBM 
based machine learning classifier which can,</b>

* Predict whether a code change request will be merged with on average 85% AUC score as soon as the
change request is submitted and assigned to a reviewer. Improving the state-of-the-art [[1]](#1) 
  by 19-27%.
* Even for new authors, predict merge probability with on average 78% AUC score. Improving the state-of-the-art [[1]](#1) 
  by 21-31%.
* Provides two adjusted approaches to update prediction for each new revision of the same change request,
  and maintains significant results. 
  
* Complete training within several seconds, hence feasible to use in real world projects.
  

# Project Structure 
* Data :
  
  Each project directory contains their raw data, features and experimentation 
  results.
    * Eclipse
    * Gerrithub
    * Libreoffice
* Code
    * <b>Complete Mining Process</b>: Contains complete raw change data mining steps 
      (except file diff).
      
    * <b>config</b> : Basic configuration for data path and models.
    * <b>Feature calculator</b>: Calculates feature sets from raw data created after
    mining.
    * <b>Feature calculator for Fan</b>: Calculates feature for state-of-the-art work by
    Fan et al.[[1]](#1). Their shared repository can be found [here](https://github.com/YuanruiZJU/EarlyPredictionReview).
      
    * <b>Feature calculator for multiple revisions</b>: Calculates features when prediction 
    is updated for each new revision of the change request.
      
    * <b>Longitudinal 10 fold cross validation</b>: Runs each of the experiments presented is our
    paper with longitudinal cross validation setup.
        
    * <b>Longitudinal 10 fold cross validation - Fan</b>: Runs each of the experiments for Fan's [[1]](#1)
    work we have compared in our paper, with longitudinal cross validation setup.
  
    * <b>Mine file diff </b>: Mines file diff data for first revision of each selected code changes.
    Used later to calculated code segment related features.
  
    * <b>Miner</b>: Contains the miner class implementation, used to mine code changes from Gerrit.
  * <b> SimpleParser </b>: Parses the json responses from Gerrit server and return them in Class.
    *<b> Util</b>: Contains some util methods.
    
# How to run
Open `Predict-Code-Change` as project in Pycharm or any other python IDE. People interested in testing the 
tool should directly jump to [Experimentation](#exp) section. All the data need for running the experiments are
already uploaded. If you want to run them on your own mined dataset, complete the following two sections first.
## Mining
* Run the `Complete Mining Process.py` file to start mining.
* Set the project name, make sure Gerrit class has corresponding Gerrit server address for this project.
* Check if the directories for the data to be dumbed is created.
* Set the start and end time period, changes created and closed in that period will be collected. Make
  sure the time format is valid. Check existing code in `Miner.py` for example.
* This step is long and time-consuming. Specially, because during downloading large chunks of 
data, Gerrit servers randomly close the connection, and you have to rerun the miner several times
till it is successful in mining all changes within a period.

* <b> For best experience, run each steps in the miner individually. </b> When you want 
  to run one,comment out the others.
  
Gerrit change response collected by `Miner.py` doesn't contain file contents.
* Run the `Mine file diff.py` after completing previous steps, to mine
file contents for first revision of each selected change request.
* This miner doesn't batch download, so expect a looooong time to finish 
mining changes. Also, occasionally Gerrit will close connections or send
  response not found messages. Rerunning the miner might fix that issue sometimes.
  

* With mined data project structure will look like this for each project
  * Eclipse
    * change : Batch of change requests
    * changes: Individual change requests.
    * diff: File diff content for first revision of each change request.
    * profile: Profile of Gerrit authors.

## Feature calculation
* This step can run only after completing previous mining steps.
* Currently, our raw data isn't added here.
* Run `Feature calculator.py` to calculate features from raw data. It sorts selected changes 
from `Project_selected_change_list.csv` and searches for their corresponding mined files to calculate feature.
* So for each change in the selected list, the followings must be present before 
  calculating feature. An example is present in Eclipse project.
  * Project
    * changes
      * Project_changeNumber_change.json
    * diff
      * Project_changeNumber_diff.json
    * profile
      * profile_accountId.json
  


## <a id="exp">Experimentation</a> 
* Set config values in file `config.py`. For example project, number of runs, folds, 
  feature list, data path, seed.
  
* Running `Longitudinal 10 fold cross validation.py` with run our model experiments for `project` 
  as specified in `config.py` file, `runs` times, using `folds` number of folds. It will generate
  the following files in `Data\project` folder :
    * <b>project_train_result_cross.csv </b>: Average train performance for each fold.
    *<b> project_test_result_cross.csv </b>:Average test performance for each fold.
    * <b>project_result_cross.csv</b>: Overall average performance across folds.
    * <b>project_feature_importance_cross.csv</b>: Average feature importance calculated from LightGBM
    `feature_importances_` attribute.
  
* Similarly, running `Longitudinal 10 fold cross validation - Fan.py` with run Fan's [[1]](#1) 
  model experiments for parameters specified in `config.py` file.

# References
<a id="1">[1]</a> 
[Y. Fan, X. Xia, D. Lo, S. Li, Early prediction of merged code changes to prioritizereviewing tasks, 
Empirical Software Engineering (2018) 1–48.](https://link.springer.com/content/pdf/10.1007/s10664-018-9602-0.pdf)


