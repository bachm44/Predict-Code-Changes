# Data mining
## Gerrit Miner
* For each project mine change details within a time period.
* Create a list of change_ids and account_ids from mined data.
* Use account list to mine profile info and profile history.
* Use change_ids list to mine comments.

Use Complete Mining Process.py for this.
This will 
* mine all changes with in a certain time period for a project
* then create a summary file called <b>change_list.csv</b>
* then make a list of account_ids <b>account_list.csv</b>
* download comments for each change
* download account registration info (join date ) and their work details
 
 ## Create registration info
 Extract joindates from profile_account_id.json files using <b>Extract join dates.py </b>
 and dump that into <b> joindates.csv </b>
 ## Create work details
 Extract work details for each account id from profile_details_account_id.json using
 <b> Extract profile details.py </b> and dump that into <b>profile_details.csv </b>.
 
## Feature Extraction using parser
In case of closed changes "mergeable": true and "work_in_progress": true is only 
available for abandoned changes. "submit_type" is a field whose value changes after being merged.
So open and abandoned changes have it, when merged changes don't. Merged changes have
'submitted' column. Where other changes don't.

For this section we'll use csv files created in previous sections using <b> Feature
calculator from files.py </b>
