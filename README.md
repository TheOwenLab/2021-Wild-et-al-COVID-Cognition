# Seeing through brain fog: disentangling the cognitive, physical, and mental health sequelae of COVID-19.

#### [Conor J. Wild](cwild@uwo.ca)<sup>1,2*</sup>, Loretta Norton<sup>1,3</sup>, David K. Menon<sup>4</sup>, David A. Ripsman<sup>5</sup>, Richard H. Swartz<sup>6+</sup>, & Adrian M. Owen<sup>1,7+</sup>

1.	The Brain & Mind Institute, Western University, London, Canada
2.	Physiology and Pharmacology, Western University, London, Canada
3.	Department of Psychology, King's University College, Western University, London, Canada
4.	Division of Anaesthesiology, Addenbrooke's Hospital, University of Cambridge, Cambridge, UK
5.	Department of Medicine, University of Ottawa, Ottawa, Canada
6.	Department of Medicine (Neurology), Hurvitz Brain Sciences Program, Sunnybrook HSC, University of Toronto, Toronto, Canada
7.	Department of Psychology, Western University, London, Canada

## v0.5 - Revisions (1st round)

## Abstract
As COVID-19 cases exceed hundreds of millions globally, many survivors face cognitive challenges and prolonged symptoms. However, important questions about the cognitive impacts of COVID-19 remain unresolved. In the present online study, 478 volunteers who self-reported a confirmed COVID-positive test completed a comprehensive cognitive battery and an extensive questionnaire. This group performed significantly worse than pre-pandemic controls on cognitive measures of reasoning, verbal, and overall performance, and processing speed, but not short-term memory – suggesting domain-specific deficits. We identified two distinct factors underlying health measures: one varying with physical symptoms and illness severity, and one with mental health. Crucially, within the COVID-positive group cognitive performance was correlated with physical symptoms, but not mental health, and significant impairments were evident even in the subgroup that did not require hospitalisation. These findings suggest that the subjective experience of “long COVID” or “brain fog” relates to a combination of physical symptoms and cognitive deficits.

## Data Availability
All anonymized data (i.e., cognitive testing and survey/questionnaire data) collected for this study are publicly available in the Scholar Portal Dataverse, as a part of this record (https://doi.org/10.5683/SP2/ZQR9QQ), and can be used to recreate all statistics, tables, and figures in this manuscript.

## Preprint
https://www.researchsquare.com/article/rs-373663/v1
Note, some of the tables in the preprint linked here need some help / reformatting. For better quality, check out the PDFs included in the preprint directory here.

## Files

### covid_cognition.ipynb
A viewable Python notebook of the entire analysis, including embedded images. It represents the most recent version of the analysis, including all tables, figures, and numbers. If you are unable to view the notebook in GitHub by clicking the file above (e.g., you get the _"Sorry, something went wrong."_ message), trying opening it at [nbviewer](https://nbviewer.jupyter.org/github/TheOwenLab/2021-Wild-et-al-COVID-Cognition/blob/master/covid_cognition.ipynb?flush_cache=True).

###  covid_cognition.ipy
The executable analysis script used to generate the notebook.

### ./cbs_data/
A subpackage for loading data from this study. Download data files from the [repository](https://doi.org/10.5683/SP2/ZQR9QQ), and place them in this `./cbs_data/data`.

### ./covid_cognition/
A subpackage that contains miscellaneous helpers and functions for performing statistics, generating figures, etc.

### requirements.txt
Contains the frozen list of python packages (and their versions) used in this data analysis.

### ./preprint/
Contains PDF files of the manuscript preprint.


## Running This Analysis

1. You must have a working Python (>v3.7) development environment, and probably a new (empty) [virtual environment](https://virtualenvwrapper.readthedocs.io/en/latest/).

1. Clone this repository

1. Install the required packages: `pip3 install -r requirements.txt`

1. Download the [data files](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/ZQR9QQ) and place them in the `./cbs_data/data/` folder.

1. Run the script `./covid_cognition.py`

#
