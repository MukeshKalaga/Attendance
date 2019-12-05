from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import cv2, sys, numpy, os, json,pickle,face_recognition
from sklearn import svm

main_path = os.path.join(settings.BASE_DIR,"attendance/static")
haar_file = os.path.join(main_path,'haarcascade_frontalface_default.xml')
datasets = os.path.join(main_path,"dataset")


# print(os.path.isfile(datasets))
(width, height) = (130, 100)	 
face_cascade = cv2.CascadeClassifier(haar_file) 

@csrf_exempt
def index(request):
    if request.method == 'POST':
        fileup = request.FILES["image"]
        name = request.POST.get('name')
        path = os.path.join(datasets, name) 
        if not os.path.isdir(path): 
            os.mkdir(path)
        n=1
        while os.path.isfile(os.path.join(path,"1.jpg")) or os.path.isfile(os.path.join(path,str(n)+".png")):
            n+=1
        data = fileup.read()
        image = numpy.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image,cv2.IMREAD_COLOR)
        im = image
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (width, height)) 
            cv2.imwrite('% s/% s.png' % (path, n), face_resize) 
        return HttpResponse("Thank you")
    return HttpResponse("Hello, world.")

@csrf_exempt
def recog(request):
    fileup = request.FILES["image"]
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    (width, height) = (130, 100) 
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]
    model = cv2.face.LBPHFaceRecognizer_create() 
    model.train(images, lables) 
    recface = []
    data = fileup.read()
    image = numpy.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    im = image
    imht,imwd, imdep = im.shape
    rath = imht/100
    ratw = imwd/100
    while imwd>1080 or imht>760:
        imht-=rath
        imwd-=ratw

    im = cv2.resize(im, (int(imwd),int(imht))) 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
        # Try to recognize the face 
        prediction = model.predict(face_resize) 
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
        if prediction[1]<300:
            recface.append(names[prediction[0]]) 
    # Create a Numpy array from the two lists above 
    return JsonResponse(recface,safe=False)

@csrf_exempt
def recog2(request):
    fileup = request.FILES["image"]
    data = fileup.read()
    image = numpy.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    test_image = image
    recface = []
    face_locations = face_recognition.face_locations(test_image)
    clf = pickle.load(open(main_path+"/finalized_model.sav", 'rb'))
    no = len(face_locations)
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        recface.append(*name)
    return JsonResponse(recface,safe=False)
