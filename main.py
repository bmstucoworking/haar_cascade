import urllib2
import urllib
import cv2
import numpy as np
import os


def store_raw_images():
    neg_images_link = '//image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893'
    request = urllib2.Request("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893")
    neg_image_urls = urllib2.urlopen(request).read()
    pic_num = 1

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n')[-100:]:
        try:
            print(i)
            urllib.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)

def quick_resize():
    img = cv2.imread('rsz_target.jpg')
    resized = cv2.resize(img,(50,50))
    cv2.imwrite('target_resized.jpg',resized)
# store_raw_images()
# find_uglies()
# quick_resize()
# create_pos_n_neg()

cup_cascade = cv2.CascadeClassifier('data/cascade.xml')

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # add this
    # image, reject levels level weights.
    watches = cup_cascade.detectMultiScale(gray, 1.3, 25) # SCALE FACTOR, kol-vo sosednih kadrov

    # add this
    for (x, y, w, h) in watches:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # print "detected"
    out.write(img)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# opencv_createsamples -img rsz_cup.jpg -bg bg.txt -info info/info2.lst info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 600
# opencv_createsamples -info info/info2.lst -num 600 -w 20 -h 20 -vec positives.vec
# opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 550 -numNeg 300 -numStages 20 -w 20 -h 20

