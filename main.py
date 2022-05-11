#import statements
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from random import randint
import warnings
warnings.filterwarnings("ignore")
#import various machine learning algorithms
#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
#svm
from sklearn import svm
#knn
from sklearn.neighbors import KNeighborsClassifier
#lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

####################################################
### EDIT THESE GLOBAL VARIABLES TO ALTER RESULTS ###
####################################################
TEST_SPLIT = 0.20
CORRECTNESS_WINDOW = 7.5
ITERATIONS = 100000
#note this value only applies to KNN model
NUM_KNN_NEIGHBORS = 5
#affects printed results in terminal
PRINT_INDIVIDUAL_ACC_RATES = False
#sleep score that you are aiming for each night
#1-100, 75 is considered average by Muse
GOAL = 90
#number of random predictions generated
NUM_PREDICTIONS = 100000
#precision of bed and wake times and sleep duration in minutes
#(how precise do you want your times to be?)
BED_WINDOW = 15
WAKE_WINDOW = 15
DURATION_WINDOW = 15


print("\n\nITERATIONS:", ITERATIONS)
print("TRAIN-TEST SPLIT:", str((1-TEST_SPLIT)*100) + "-" + str(TEST_SPLIT*100))
print("CORRECTNESS WINDOW:", CORRECTNESS_WINDOW)
print("NUM PREDICTIONS:", NUM_PREDICTIONS)

####################################################
####################################################
####################################################

def apply_model(model_name, X, y):

    #establish train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

    #record start time
    start = time.time()

    #apply model
    if(model_name == "decision tree classifier"):
        model = DecisionTreeClassifier()
    elif(model_name == "svm"):
        model = svm.SVC(kernel='linear')
    elif(model_name == "knn"):
        model = KNeighborsClassifier(n_neighbors=NUM_KNN_NEIGHBORS)
    elif(model_name == "lda"):
        model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    #make predictions
    predictions = model.predict(X_test)

    #calculate accuracy rate
    correct = 0
    for i in range(0, len(predictions)):
        #if prediction is in the window, it is correct
        if predictions[i] + CORRECTNESS_WINDOW >= y_test.values[i] and predictions[i] - CORRECTNESS_WINDOW <= y_test.values[i]:
            correct += 1
        #otherwise, it is wrong
    accuracy_rate = correct / len(predictions)

    #end timing
    end = time.time()

    if PRINT_INDIVIDUAL_ACC_RATES:
        print("" + model_name + " accuracy rate: " + str(accuracy_rate * 100) + '%')
        print("Time elapsed [s] " + str(end - start) + '\n')

    return accuracy_rate

def best_model(dtc_avg, svm_avg, knn_avg, lda_avg):

    models = []
    models.append(("dtc", dtc_avg))
    models.append(("svm", svm_avg))
    models.append(("knn", knn_avg))
    models.append(("lda", lda_avg))

    best_score = -1
    best_index = -1
    
    for i in range(0,4):
        if models[i][1] > best_score:
            best_score = models[i][1]
            best_index = i

    return models[best_index][0]

def create_best_model(best_model_name, X, y):

    if(best_model_name == "dtc"):
        model = DecisionTreeClassifier()
    elif(best_model_name == "svm"):
        model = svm.SVC(kernel='linear')
    elif(best_model_name == "knn"):
        model = KNeighborsClassifier(n_neighbors=NUM_KNN_NEIGHBORS)
    elif(best_model_name == "lda"):
        model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    return model

def get_goal_answers(data, best_model_name, X, y):

    #create instance of "best model"
    model = create_best_model(best_model_name, X, y)

    #create predictions using a massive number of random tests
    #array to store bed time, wake time, and sleep duration for scores >= goal
    answers = []
    for i in range(0,NUM_PREDICTIONS):
        
        #assign realistic yet random values for each feature
   
        day = randint(data["day"].min(), data["day"].max())
        month = randint(data["month"].min(), data["month"].max())
        year = randint(data["year"].min(), data["year"].max())
        mins_start_time = randint(int(data["mins_start_time"].min()), int(data["mins_start_time"].max()))
        mins_end_time = randint(int(data["mins_end_time"].min()), int(data["mins_end_time"].max()))
        mins_time_in_bed = randint(int(data["mins_time_in_bed"].min()), int(data["mins_time_in_bed"].max()))
        mins_time_asleep = randint(int(data["mins_time_asleep"].min()), int(data["mins_time_asleep"].max()))
        awake = randint(int(data["awake"].min()), int(data["awake"].max()))
        rem = randint(int(data["rem"].min()), int(data["rem"].max()))
        light = randint(int(data["light"].min()), int(data["light"].max()))
        deep = randint(int(data["deep"].min()), int(data["deep"].max()))
        deep_sleep_intensity = randint(int(data["deep_sleep_intensity"].min()), int(data["deep_sleep_intensity"].max()))
        upright = randint(int(data["upright"].min()), int(data["upright"].max()))
        left = randint(int(data["left"].min()), int(data["left"].max()))
        back = randint(int(data["back"].min()), int(data["back"].max()))
        right = randint(int(data["right"].min()), int(data["right"].max()))
        front = randint(int(data["front"].min()), int(data["front"].max()))
        avg_heart_rate = randint(int(data["avg_heart_rate"].min()), int(data["avg_heart_rate"].max()))
        active = randint(int(data["active"].min()), int(data["active"].max()))
        relaxed = randint(int(data["relaxed"].min()), int(data["relaxed"].max()))

        #make prediction
        p = model.predict([[day, month, year, mins_start_time, mins_end_time, mins_time_in_bed, mins_time_asleep, awake, rem, light, deep, deep_sleep_intensity, upright, left, back, right, front, avg_heart_rate, active, relaxed]])
        prediction = p[0]

        #add if prediction >= goal
        if prediction >= GOAL:
            answers.append((mins_start_time, mins_end_time, mins_time_asleep))

    return answers

def mean_results(answers):

    avg_bed = 0
    avg_wake = 0
    avg_duration = 0
    size = len(answers)
    
    for i in range(0,size):
        avg_bed += answers[i][0]
        avg_wake += answers[i][1]
        avg_duration += answers[i][2]

    avg_bed /= size
    avg_wake /= size
    avg_duration /= size

    return (avg_bed, avg_wake, avg_duration)

def median_results(answers):

    med_bed = answers[int(len(answers)/2)][0]
    med_wake = answers[int(len(answers)/2)][1]
    med_duration = answers[int(len(answers)/2)][2]

    return (med_bed, med_wake, med_duration)

def print_answer(stats):

    hour = 0
    minute = 0

    #bed time
    #after midnight
    if(stats[0] >= 0):
        hour = int(stats[0]//60)
        if hour == 0:
            hour = 12
        minute = int(stats[0]%60)
        min_string = str(minute)
        if(len(min_string) == 1):
            min_string = '0' + min_string
        print("\nOptimal bed time: " + str(hour) + ":" + min_string + " AM +/- " + str(BED_WINDOW) + " minutes")
    else:
        hour = int((24*60 + stats[0])//60 - 12)
        minute = int((24*60 + stats[0])%60)
        min_string = str(minute)
        if(len(min_string) == 1):
            min_string = '0' + min_string
        print("Optimal bed time: " + str(hour) + ":" + min_string + " PM +/- " + str(BED_WINDOW) + " minutes")

    #wake time
    hour = int(stats[1]//60)
    minute = int(stats[1]%60)
    #waking up in PM
    min_string = str(minute)
    if(len(min_string) == 1):
        min_string = '0' + min_string
    if(hour%24 >= 13):
        hour -= 12
        print("Optimal wake time: " + str(hour) + ":" + min_string + " PM +/- " + str(WAKE_WINDOW) + " minutes")
    elif hour == 12:
        print("Optimal wake time: " + str(hour) + ":" + min_string + " PM +/- " + str(WAKE_WINDOW) + " minutes")
    else:
        print("Optimal wake time: " + str(hour) + ":" + min_string + " AM +/- " + str(WAKE_WINDOW) + " minutes")

    #sleep duration
    hour = int(stats[2]//60)
    minute = int(stats[2]%60)
    print("Optimal sleep duration: " + str(hour) + " hours and " + str(minute) + " minutes +/- " + str(DURATION_WINDOW) + " minutes")

def main():
    
    #record start time
    start = time.time()
    
    #set up data
    data = pd.read_excel('sleep_data/converted_data.xlsx')
    X = data.drop(columns=['score'])
    y = data['score']

    #Run each model
    dtc_avg = 0
    svm_avg = 0
    knn_avg = 0
    lda_avg = 0
    for i in range (0, ITERATIONS):
        #Decision Tree Classifier
        dtc_avg += apply_model("decision tree classifier", X, y)
        svm_avg += apply_model("svm", X, y)
        knn_avg += apply_model("knn", X, y)
        lda_avg += apply_model("lda", X, y)
    #compute average
    dtc_avg/=ITERATIONS
    svm_avg/=ITERATIONS
    knn_avg/=ITERATIONS
    lda_avg/=ITERATIONS

    #print averages
    print("\nAVERAGE DTC:", str(dtc_avg * 100) + '%')
    print("AVERAGE SVM:", str(svm_avg * 100) + '%')
    print("AVERAGE KNN:", str(knn_avg * 100) + '%')
    print("AVERAGE LDA:", str(lda_avg * 100) + '%')

    #determine which algorithm is best
    best_model_name = best_model(dtc_avg, svm_avg, knn_avg, lda_avg)
    print("Highest scoring model: " + best_model_name.upper())

    #get all the bed times, wake times, and sleep durations
    # that correspond with sleep scores >= GOAL
    answers = get_goal_answers(data, best_model_name, X, y)

    #find reasonable window of bed times, wake times, and sleep durations
    #mean
    mean = mean_results(answers)
    median = median_results(answers)

    #print("mean:", mean)
    #print("median:", median)

    #convert values into readable printing format
    #mean
    print("Results using mean: \n")
    print_answer(mean)
    print("Results using median: \n")
    print_answer(median)

    #record end time
    end = time.time()

    #print total run time
    print("\nTOTAL RUN TIME [s]: " + str(end - start) + '\n')



if __name__ == '__main__':
	main()