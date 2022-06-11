# Determining Optimal Sleeping Schedules for Teenagers

## Overview

The source code is included to allow for the reproducibility of our results. Make sure to download both the main.py file, as well as the dataset. 

## Getting Started

The code for this project can be accessed via the GitHub repository. The dataset can be downloaded from Google Drive using the following link:
https://docs.google.com/spreadsheets/d/17iekB7YOnXmgsQI9F3disB0UMmPV1g3O/edit?usp=sharing&ouid=104470998691592532088&rtpof=true&sd=true

## Prerequisites 

The code uses Python 3.8. Additionally, make sure to install the following packages using pip or pip3 in the terminal.

```
$ pip3 install pandas
$ pip3 install sklearn
```

## Recreate Results
To recreate the results, ensure that the dataset (converted_data.xlsx) is put inside a folder (which should be named "sleep_data"). Make sure that the code (main.py) is in the same folder that the folder sleep_data is in. To run the code to recreate the results, simply navigate to the project folder and type the following in the terminal:

```
python3 main.py
```

## Alter Results
There are a number of variables that can be altered to change the results. These variables are global variables, and are defined in all caps directly below the import statements.

- TEST_SPLIT
- CORRECTNESS_WINDOW
- ITERATIONS
- NUM_KNN_NEIGHBORS
- PRINT_INDIVIDUAL_ACC_RATES
- GOAL
- NUM_PREDICTIONS
- BED WINDOW
- WAKE WINDOW
- DURATION WINDOW


TEST_SPLIT: the value corresponding to the test portion of the train test split when training the four machine learning algorithms. This value is originally set to 0.2, making the train split 0.8.

CORRECTNESS_WINDOW: refer to our paper to understand the correctness window.

ITERATIONS: the number of times each machine learning algorithm is applied to the dataset (used to determine which of the four algorithms is most effective on the dataset)

NUM_KNN_NEIGHBORS: the number of neighbors used in the KNN algorithm 

PRINT_INDIVIDUAL_ACC_RATES: prints additional information in terminal, recommended to leave as False

GOAL: minimum sleep score (1-100) that subject intends to aim for for each night of sleep. Refer to paper for more information.

NUM_PREDICTIONS: number of sample nights of sleep generated. Refer to paper for more information. 

BED_WINDOW, WAKE WINDOW, DURATION_WINDOW: these windows are added to each of the result values to provide some room for error and logistical freedom. In other words, it is difficult to go to bed exactly at the minute that is recommended by the algorithm, so this window provides a logistic buffer for the general window of time that is best for the individual. 

## Authors

* **Zach Gillette** - *Initial work* - [ztgillette](https://github.com/ztgillette)

* **Orlando Azuara** - *Initial work* - [orlandoazu0709](https://github.com/orlandoazu0709)
