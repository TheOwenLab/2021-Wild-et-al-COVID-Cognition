# Seeing through brain fog: disentangling the cognitive, physical, and mental health sequelae of COVID-19.

## NOTE: This repository is still currently under construction, expect significant changes over the next weeks.

## Abstract
As COVID-19 cases exceed hundreds of millions globally, it is clear that many survivors face cognitive challenges and prolonged symptoms. However, important questions about the cognitive impacts of COVID-19 remain unresolved. In the present online study, 485 volunteers who reported having had a confirmed COVID-positive test completed a comprehensive cognitive battery and an extensive questionnaire. This group performed significantly worse than pre-pandemic controls on cognitive measures of reasoning, verbal, and overall performance, and processing speed, but not short-term memory – suggesting domain-specific deficits. We identified two distinct factors underlying health measures: one varying with physical symptoms and illness severity, and one with mental health. Crucially, cognitive deficits were correlated with physical symptoms, but not mental health, and were evident even in cases that did not require hospitalisation. These findings suggest that the subjective experience of “long COVID” or “brain fog” relates to a combination of physical symptoms and cognitive deficits.

## Data Availability
All anonymized data (i.e., cognitive testing and survey/questionnaire data) collected for this study are publically avaiable in the Scholar Portal Dataverse, as a part of this record: https://doi.org/10.5683/SP2/ZQR9QQ. Data from the normative (control) sample are not publically available at this time due to ethical constraints under which they were collected. However, they are available upon request from the corresponding author, (Conor J. Wild)[cwild@uwo.ca]. Soon, they will be avaiable via request to Western University Dataverse (Link to appear, stay tuned!).

## Preprint
https://www.researchsquare.com/article/rs-373663/v1
Note, some of the tables in the preprint linked here need some help / reformatting. For better quality, check out the PDFs included in the preprint directory here.

## Files

### covid_cognition.ipynb
A viewable Python notebook of the entire analysis, including embedded images. It represents the most recent version of the analysis, including all tables, figures, and numbers. If you are unable to view the notebook in GitHub by clicking the file above (e.g., you get the _"Sorry, something went wrong."_ message), trying opening it at [nbviewer](https://nbviewer.jupyter.org/github/TheOwenLab/2021-Wild-et-al-COVID-Cognition/blob/master/covid_cognition.ipynb?flush_cache=True).

### ./covid_cognition/
This folder contains all the custom source code written for, and needed to recreate, all analyses in this study.

### ./preprint/
Contains PDF files of the manuscript preprint.

### ./data/
Download the two .CSV files from the (repository)[https://doi.org/10.5683/SP2/ZQR9QQ], and place them in this folder.

### requirements.txt
Contains the frozen list of python packages (and their versions) used in this data analysis.

## Running This Analysis

1. You must have a working Python (>v3.7) development environment, and probably a new (empty) [virtual environment](https://virtualenvwrapper.readthedocs.io/en/latest/).

1. Clone this repository

1. Install the required packages: `pip3 install -r requirements.txt`

1. Download the data files (links to come), place them in the `./data/` folder in the root of the repo.

1. Run the script `./covid_cognition/covid_cognition.py`

#
