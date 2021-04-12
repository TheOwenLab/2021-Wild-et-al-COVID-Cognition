# Seeing through brain fog: disentangling the cognitive, physical, and mental health sequelae of COVID-19.

## NOTE: This repository is still currently under construction, expect significant changes over the next weeks.

## Abstract
As COVID-19 cases exceed hundreds of millions globally, it is clear that many survivors face cognitive challenges and prolonged symptoms. However, important questions about the cognitive impacts of COVID-19 remain unresolved. In the present online study, 485 volunteers who reported having had a confirmed COVID-positive test completed a comprehensive cognitive battery and an extensive questionnaire. This group performed significantly worse than pre-pandemic controls on cognitive measures of reasoning, verbal, and overall performance, and processing speed, but not short-term memory – suggesting domain-specific deficits. We identified two distinct factors underlying health measures: one varying with physical symptoms and illness severity, and one with mental health. Crucially, cognitive deficits were correlated with physical symptoms, but not mental health, and were evident even in cases that did not require hospitalisation. These findings suggest that the subjective experience of “long COVID” or “brain fog” relates to a combination of physical symptoms and cognitive deficits.

## Files

### 1. covid_cognition.ipynb
A viewable notebook of the entire analysis, including embedded images. Represents the most recent version of the analysis, including all tables, figures, and numbers. If you are unable to view the notebook in GitHub by clicking the file above (e.g., you get the _"Sorry, something went wrong."_ message), trying opening it at [nbviewer](https://nbviewer.jupyter.org/).

### 2. covid_cognition.py
The actual python script that I run to perform the analysis. The notebook file (above) is generated from the output of this script.

### 3. lib_stats.py
Contains res-usable functions for performing statistical analyses. They exist in this file, instead of in the notebook, to increase code readability, re-useability, and maintainability.

### 4. lib_plots
Similar to #3, but for making plots.

### 5. lib_colours
Again, helper functions and constants for defining and working with colours.

### 6. lib_chord
Contains code for generating the chord plots used to visualize the factor analyses (Figure 1) in this study.
