import numpy as np
import cv2
from sklearn.covariance import EmpiricalCovariance
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import os

def extract_sift_brief(image):
    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(image, None)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(image, keypoints)
    cov = EmpiricalCovariance().fit(descriptors).covariance_
    cov = cov.ravel()

    output_image = cv2.drawKeypoints(
        image, 
        keypoints, 
        None,  # Keypoint'ler çizileceği yeni bir görüntü oluşturur
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    cv2.imwrite("keypoints.jpg", output_image)


    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptors = brief.compute(image, keypoints)
    cov_1 = EmpiricalCovariance().fit(descriptors).covariance_
    cov_1 = cov_1.ravel()

    fea = cov.tolist() + cov_1.tolist()
    return np.array(fea)

def read_test_labels(filename):
    df = pd.read_csv(filename)
    print(df.head())
    return df


def prepare_data(data_dir, extract_func = extract_sift_brief):
    X = []
    y = []
    for fn in os.listdir(data_dir):
        sl = fn.split(".")
        sl_1 = sl[0].split("_")
        label = int(sl_1[0])
        print(fn)
        filename = os.path.join(data_dir, fn)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        x = extract_func(img)
        X.append(x)
        y.append(label)
    return np.array(X), np.array(y)

def prepare_data_test(test_dir, extract_func = extract_sift_brief):
    X = []
    for fn in os.listdir(test_dir):
        filename = os.path.join(test_dir, fn)
        print(fn)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        x = extract_func(img)
        X.append(x)
    return X


if __name__ == "__main__":
    X, y = prepare_data("../../../data/awic2012/images/training")
    X_test = prepare_data_test("../../../data/awic2012/images/test")
    #df = read_test_labels("../../../data/awic2012/solution.csv")
    #y_test = df["writer"].to_numpy()
    print(X.shape)
    sc = MaxAbsScaler()
    X = sc.fit_transform(X)
    X_test = sc.transform(X_test)
    clf = LogisticRegression(C = 0.1)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    with open("submission.h", "w") as ofstr:
        ofstr.write("int answers[] = {")
        for i, pred in enumerate(y_pred):
            if (i % 21 == 0):
                ofstr.write("\n")
            ofstr.write("%d, " % pred)
        ofstr.write("};") 
    print("Submission data is written to submission.h file")
    #print(accuracy_score(y_test, y_pred))
    



    
    
    



































#    output_image = cv2.drawKeypoints(
#    image, 
#    keypoints, 
#    None,  # Keypoint'ler çizileceği yeni bir görüntü oluşturur
#    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
#    )
#    cv2.imwrite("keypoints.jpg", output_image)
