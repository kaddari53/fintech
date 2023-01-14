import os
os.environ['TF_CPP_MIN_LOG_pip LEVEL'] = '3'
from flask import Flask
import os
from time import time
from flask import render_template,url_for,request,redirect,make_response,Response
from flask import send_from_directory, abort, session, flash, stream_with_context
import ssl
import time
import playsound
from datetime import datetime
import cv2 
import keras
#from sklearn.metrics import confusion_matrix
#for broken data stream error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
#import matplotlib.pyplot as plt
#%matplotlib inline
from keras.utils import load_img, img_to_array
import numpy as np
import smtplib
import random
import threading
from email.message import EmailMessage
import requests
import json

import pandas as pd
import numpy as np
import itertools
import requests
import json
from datetime import datetime
# from newspaper import Article
# import nltk

import random


# nltk.download('punkt')
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
random.seed()
id_to_category={0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
Pkl_Filename = "Pickle_RL_Model.pkl"  
df = pd.read_csv('BBC News Train.csv')

features = tfidf.fit_transform(df.Text).toarray()
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()        

#     def get_frame(self):
#         ret, frame = self.video.read()

#         # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

#         ret, jpeg = cv2.imencode('.jpg', frame)

#         return jpeg.tobytes()

# class Camera(object): 
#         while True:
#             # Check input queue is not full
#             if not input_q.full():
#             # Read frame and store in input queue
#                ret, frame = vs.read()
#             if ret: 
#                 input_q.put((int(vs.get(cv2.CAP_PROP_POS_FRAMES)),frame))
#                 frame = input_q.get()
#                 frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
#                 output_q.put((frame[0], detect_objects(frame_rgb, sess, detection_graph)))
#                 if not output_q.empty():
#                     # Recover treated frame in output queue and feed priority queue
#                     output_pq.put(output_q.get())
app=Flask(__name__)            
import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1
# sub = cv2.createBackgroundSubtractorMOG2()
# model = keras.models.load_model('raks_model14.h5')
# model.make_predict_function() 

def model_predict(img_path, model):


 
    preprocessed_image = prepare_image(img_path)
    predictions = model.predict(preprocessed_image)
    labels=(predictions>0.5).astype(np.int)
    
    return labels
import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals



def send_mail_function():
    email_sender="tayyaratoumaima1999@gmail.com"
    email_possword="sjph znmj fwiz crrx"
    email_receiver=[" kaddari53@gmail.com","n.abdelouali@edu.umi.ac.ma","dahhassicharifa@gmail.com"]
    subject="Fire detection & SST"
    body="""
    Warning A Fire Accident has been reported on X Company
    """
    em=EmailMessage()
    em['Form']=email_sender
    em['To']=email_receiver
    em['subject']=subject
    em.set_content(body)
    context= ssl.create_default_context()
    try: 
        with smtplib.SMTP_SSL('smtp.gmail.com', 465,context=context)as smtp:
            smtp.login(email_sender,email_possword)
            for email in email_receiver:
                smtp.sendmail(email_sender,email,em.as_string())
            smtp.close()
    except Exception as e:
    	print(e)

# def send_mail_function():

#     recipientEmail = "tayyaratoumaima1999@gmail.com"
#     recipientEmail = recipientEmail.lower()

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.ehlo()
#         server.starttls()
#         server.login("Enter_Your_Email (System Email)", 'Enter_Your_Email_Password (System Email')
#         server.sendmail('Enter_Your_Email (System Email)', recipientEmail, "Warning A Fire Accident has been reported on X Company")
#         print("sent to {}".format(recipientEmail))
#         server.close()
#     except Exception as e:
#     	print(e)
      
# def geni(camera): 
#     while True: 
#         frame = camera.get_frame() 
#         yield (b'--frame\r\n' 
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    
def hadilkhraaw():
        count=0
        camera = cv2.VideoCapture(0)
        while True:
              success, frame = camera.read()  # read the camera frame
              if not success:
                 break
              else:
                 face_names = []
                
                 frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
                 cv2.imwrite("0%d.jpg" % count, frame) 
                #  count += 1
                 preprocessed_image = prepare_image("0%d.jpg" % count)
                 predictions = model.predict(preprocessed_image)
                 labels=(predictions>0.6).astype(np.int)
                 name = "Unknown"
                 if labels[0][0]==1:
                    name="fire"

                 else :
                    name="not fire"
                 face_names.append(name)
                 print(face_names)
                 for  name in  face_names:
                        top = 40
                        right = 40
                        bottom = 50
                        left = 40


                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (6, 3, 0), 1)
                
                 ret, buffer = cv2.imencode('.jpg', frame)
                 frame = buffer.tobytes()
                 yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


















@app.route("/")
def index():
    return render_template('public/oumaima.html')
# @app.route("/jinja")
# def jinja():
#     my_name="oumaima"
#     hello ="<script>alert('you got hacked')</scripts>"



#     return render_template('public/jinja.html',my_name= my_name,hello=hello)

# def op():

#     """Video streaming generator function."""
#     cap = cv2.VideoCapture(0)
#     count=0

#     # Read until video is completed
#     while(cap.isOpened()):
#         ret, frame = cap.read()  # import image
#         if not ret: #if vid finish repeat
#             frame = cv2.VideoCapture(0)
#             continue
#         if ret:  # if there is a frame continue with code
#             image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
#             fgmask = sub.apply(gray)  # uses the background subtraction
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
#             closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
#             opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
#             dilation = cv2.dilate(opening, kernel)
#             retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
#             contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             minarea = 400
#             maxarea = 50000
#             for i in range(len(contours)):  # cycles through all contours in current frame
#                 if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
#                     area = cv2.contourArea(contours[i])  # area of contour
#                     if minarea < area < maxarea:  # area threshold for contour
#                         # calculating centroids of contours
#                         cnt = contours[i]
#                         M = cv2.moments(cnt)
#                         cx = int(M['m10'] / M['m00'])
#                         cy = int(M['m01'] / M['m00'])
#                         # gets bounding points of contour to create rectangle
#                         # x,y is top left corner and w,h is width and height
#                         x, y, w, h = cv2.boundingRect(cnt)
#                         # creates a rectangle around contour
#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         # Prints centroid text in order to double check later on
#                         cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
#                         cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
#         #cv2.imshow("countours", image)
#         frame = cv2.imencode('.jpg', image)[1].tobytes()
#         cv2.imwrite("0%d.jpg" % count, image) 
#         preprocessed_ima                         ge = prepare_image("0%d.jpg" % count)
#         predictions = model.predict(preprocessed_image)
#         labels=(predictions>0.5).astype(np.int)
#         if labels[0][0]==1:
#             a='fire'
        # else:
        #     print("noot fire")
        # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # #time.sleep(0.1)
        # key = cv2.waitKey(20)
        # if key == 27:
        #    break
              


def gen():
       
       cap = cv2.VideoCapture(0)
       count=0
       while(cap.isOpened()):
      # Capture frame-by-frame
           ret, img = cap.read()
           if ret == True:
              img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
              frame = cv2.imencode('.jpg', img)[1].tobytes()
              cv2.imwrite("0%d.jpg" % count, img) 
              preprocessed_image = prepare_image("0%d.jpg" % count)
              predictions = model.predict(preprocessed_image)
              labels=(predictions>0.5).astype(np.int)
              if labels[0][0]==1: 
                 a="fire"
                 threading.Thread(target=send_mail_function).start()
              else:
                 a="not fire"
              
                

            
               
              yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
           else:
               break
        
@app.route("/khkh")
def khkh():
    return Response(gen1(),mimetype='multipart/x-mixed-replace; boundary=frame')
def gen1():
    camera = cv2.VideoCapture(0) 
    Alarm_Status = False
    Email_Status = False
    Fire_Reported=0
    serie=[]
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (350, 300))

        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)

        output = cv2.bitwise_and(frame, hsv, mask=mask)

        no_red = cv2.countNonZero(mask)
        frame2 = cv2.imencode('.jpg', output)[1].tobytes()
        size = 100

                   
        if int(no_red) > 1000:
           Fire_Reported = Fire_Reported + 1

   
        if Fire_Reported >= 1:
            if Email_Status == False:
                threading.Thread(target=send_mail_function).start()
                Email_Status = True
                print("seeending")
            if Alarm_Status == False:
                threading.Thread(target=play_alarm_sound_function).start()
                Alarm_Status = True


        	    


                   
                   

        
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

    		       
     

    

@app.route("/temp")
def temp():
    


   return render_template("public/temperature.html")

@app.route("/admino", methods=['GET', 'POST'])
def admino():
    if request.method =="POST":
        
        req= request.form
    
        Panneau= req.get('comp_select')
       
       

       
        
        print( Panneau)
        return redirect(url_for('jazz_index', Panneau=Panneau))



        
    return render_template('macros/essai5.html', data=[{'Panneau': 'GLE.PA'}, {'Panneau': 'PRIVATE'}, {'Panneau': 'SGE.SG'},{'Panneau': 'TENERGY.AT'},{'Panneau': 'BICEY'},{'Panneau': 'SCGLY'}]
       )



        
# @app.route("/pred")
# def prediction():
                   
              

# #     return Response(op()) 
# def hayaaa():
#     camera = cv2.VideoCapture(0) 
#     Alarm_Status = False
#     Email_Status = False
#     Fire_Reported=0
#     serie=[]
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         frame = cv2.resize(frame, (350, 300))

#         blur = cv2.GaussianBlur(frame, (21, 21), 0)
#         hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#         lower = [18, 50, 50]
#         upper = [35, 255, 255]
#         lower = np.array(lower, dtype="uint8")
#         upper = np.array(upper, dtype="uint8")

#         mask = cv2.inRange(hsv, lower, upper)

#         output = cv2.bitwise_and(frame, hsv, mask=mask)

#         no_red = cv2.countNonZero(mask)
#         frame2 = cv2.imencode('.jpg', output)[1].tobytes()
#         size = 100

                   
#         if int(no_red) > 1000:
#            Fire_Reported = Fire_Reported + 1
#         return Fire_Reported

#    datetime.now().strftime('%Y-%m %H:%M')


@app.route('/chart-data')
def chart_data():
    def generate_random_data():
        while True:
            json_data = json.dumps(
                {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'value': random.random() * 0.00015})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    response = Response(stream_with_context(generate_random_data()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

# @app.route("/Temperature")
# def Temperature():

    

#         camera = cv2.VideoCapture(0) 
#         Fire_Reported = 0
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break


#             frame = cv2.resize(frame, (350, 300))

#             blur = cv2.GaussianBlur(frame, (21, 21), 0)
#             hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#             lower = [18, 50, 50]
#             upper = [35, 255, 255]
#             lower = np.array(lower, dtype="uint8")
#             upper = np.array(upper, dtype="uint8")

#             mask = cv2.inRange(hsv, lower, upper)

#             output = cv2.bitwise_and(frame, hsv, mask=mask)
            
#             if int(no_red) > 15000:
#                 Fire_Reported = Fire_Reported + 1
#             size=60
#             x_vec = np.linspace(0,1,size+1)[0:-1]
#             y_vec = np.random.randn(len(x_vec))
#             line1 = []
#             while True:
#                 rand_val = Fire_Reported
#                 y_vec[-1] = rand_val
#                 line1 = live_plotter(x_vec,y_vec,line1)


#                 y_vec = np.append(y_vec[1:],0.0)
#         return render_template("about/webstreaming.html")
       
@app.route('/f')
def show_LABEL():


    list1=[]

    list2=[]
    list3=[]
    r=requests.get("https://newsapi.org/v2/top-headlines?country=us&apiKey=4b0c8b0f57a248c9ae933dceee517b8c")
    r.content
    data=json.loads(r.content)
    

    for i in range(10):
        news=[str(data["articles"][i]["title"])]
        news2=(data["articles"][i]["title"])
        list1.append(news2)
        
        text_features = tfidf.transform(news) 
        predictions = model.predict(text_features)
        heloo=data["articles"][i]["url"]
        bo= data["articles"][i]["description"] or data["articles"][i]["content"] 
       
        list2.append(heloo)

        list3.append(bo)

        for  predicted in predictions:
            id=id_to_category[predicted]
            print("News",i+1, ":",news,"class :",{id})
        
    res = {list1[i]: list3[i] for i in range(len(list1))}
    

    date = datetime.now()
    # random.randint(0, 105)

    
            
    return render_template('macros/input_macroshhh.html', data=data)
            
    # return render_template('macros/input_macros.html', news=list1,idi=list2,b=list3)



@app.route('/jazz', methods=['GET', 'POST'])
def jazz_index():
    # import request from Flask
    Panneau=request.args.get('Panneau', None)
    import yfinance as yf
    from datetime import datetime
    date= datetime.today().strftime('%Y-%m-%d')
    if Panneau=='GLE.PA':
        sg='GLE.PA'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    if Panneau=='BICEY':
        sg='BICEY'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    if Panneau=='SCGLY':
        sg='SCGLY'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    if Panneau=='TENERGY.AT':
        sg='TENERGY.AT'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    if Panneau=='SGE.SG':
        sg='SGE.SG'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    
    if Panneau=='PRIVATE':
        sg='PRIVATE'
        data4=yf.Ticker(sg)
        dataDf4= data4.history(period='1d',start='2022-12-1',end=date)

        dataDf4['Close'].plot()
        plt.show()
    return redirect(url_for("admino"))
     

    

                
#     return render_template("about/webstreaming.html")
# @app.route("/about")
# def about():

#     return  Response(hadilkhraaw(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame',)


def prepare_image(file):
    img_path = ''
    img = load_img(img_path + file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def play_alarm_sound_function():
	while True:
		    playsound.playsound('C:/Users/PC/Desktop/essai/app/WhatsApp-Audio-2022-11-24-at.mp3')
            #C:\Users\PC\Desktop\essai\app\WhatsApp-Audio-2022-11-24-at-12.35.20.mp3
                   



        

app.config["SECRET_KEY"]= 'dXlH4yE6HTnYyicyhOZ7Qw'


@app.errorhandler(404)
def not_found(e):
    return render_template("public/404.html")

    

users = {
    "Mohamed HOSNI": {
        "username": "Mohamed HOSNI",
        "email":"Mohamed HOSNI@gmail.com",
        "bio":"Professeur AI ENSAM Méknes",
        "password":"ensam"
    },
    "Tawfik Masrour ":{
        "username": "Tawfik Masrour ",
        "email":"Tawfik Masrour@gmail.com",
        "bio":"chef de la filière IA ENSAM Méknes",
        "password":"ensam"
    },"samir amri":{
        "username": "samir amri",
        "email":"samir amri@gmail.com",
        "bio":"Professeur AI ENSAM Méknes",
        "password":"ensam"
}}

@app.route("/sign-in", methods=["GET","POST"])
def sing_in():
 

    
    if request.method =="POST":

        req= request.form
        print("hehoooooo",req)
        username= req.get('username')
        print("hehoooooo",username)

        password = req.get('password')
        print("hehoooooo",password)
        if not username in users:
            flash("username not found","danger")
            return redirect(request.url)
        else:
            user=users[username]

        if not password == user["password"]:
            flash("Password not found","danger")
            return redirect(request.url)
        else:
            session["USERNAME"]=user["username"]
            flash("Welcome to our world","success")

            return redirect(url_for("profile1"))
        
    return render_template("public/sinin.html")

# @app.route("/", methods=["GET","POST"])
# def sign_up():

#     if request.method =="POST":
        
#         req= request.form
#         username= req.get('username')
#         email=req.get("email")
#         password = req.get('password')

#         if not len((password))>=10:
#            flash("Password must be at least 10 characters in length","danger")
#            return redirect(request.url)
#         flash("account created","success")
#         return redirect(request.url)

        

        
    # return render_template("public/sign_up.html")

# @app.route("/profile/<username>")
# def profile(username):
#       user= None
#       if username in users:
       
#            user=users[username]
      

#       return render_template("public/profile.html", user=user, username=username)


# @app.route("/multiple/<foo>/<bar>/<baz>")
# def multi(foo,bar,baz):
#     return f" foo is {foo},bar is {bar},baz is {baz}"
# @app.route("/json",methods=["POST"])
# def json():
#     if request.is_json:

       
#        req=request.get_json()
#        response={"message":"json recieved",
#        "name":req.get("name")}
#        res=make_response(jsonify(response),200)

#        return res
#     else:
#         res=make_response(jsonify({"message":"NO JSON RECEIVED"}),400)
#         return res
# @app.route("/guestbook")
# def  guestbook(): 
#      return render_template("public/guestbook.html") 
# @app.route("/guestbook/create_entry", methods=["POST"])
# def create_entry(): 
#     req= request.get_json()
#     res=make_response(jsonify(req), 200)
#     return res
# @app.route("/query")
# def query():



#     args = request.args
#     print(args)

#     return "Query received",200



app.config['IMAGE_UPLOADS'] = "C:/Users/PC/Desktop/essai/app/static/img/uploads"

app.config['ALLOWED_IMAGE_EXTENSIONS'] = ["PNG", "JPG","JPEG","GIF"]




# def allowed_image(filename):
#     if not "." in filename:
#         return False
#     ext=filename.rsplit(".",1)[1]
#     if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
#         return True
#     else:
#         return False
    

# @app.route("/upload-image", methods=["GET","POST"])
# def upload_image():
#     if request.method =="POST" :
#          if request.files:  


#               image=request.files["image"]
#             #   image_dir_name=secrets.token_hex(16)
#             #   os.mkdir(os.path.join(app.config["IMAGE_UPLOADS"],image_dir_name))
#               if image.filename == "":
#                 print("image must have a filename")
#                 return redirect(request.url)
#               if not allowed_image(image.filename):
#                 print("extention not allowed")
#                 return redirect(request.url)
#               else:
#                 filename = secure_filename(image.filename)
                 
#                 image.save(os.path.join(app.config['IMAGE_UPLOADS'],filename))
#               print("image saved")
#               return redirect(request.url)




#     return render_template("public/upload_image.html")
@app.route("/admin",methods=["GET","POST"])
def admin_dashboard():
  
    r=requests.get("https://newsapi.org/v2/top-headlines?country=us&apiKey=4b0c8b0f57a248c9ae933dceee517b8c")
    r.content
    data=json.loads(r.content)
    r=[]
    e=[]
    for i in range(20):
        news=[str(data["articles"][i]["title"])]
        
        
        
        text_features = tfidf.transform(news) 
        predictions = model.predict(text_features)
        y_pred_proba = model.predict_proba(text_features)
        

        pres=max (list(y_pred_proba))
       
        maxx=max(pres)
       
 
       
        

        for  predicted in predictions:
           

            if predicted==0:

                e.append(maxx)
                
                r.append(i)
    dict_from_list = dict(zip(e, r))
    print(dict_from_list)
    keys = dict_from_list.keys()
    sorted_keys = sorted(keys)
    print(sorted_keys)

    sorted_desserts = {}
    for key in sorted_keys:
        sorted_desserts[key] = dict_from_list[key]
    
    print(sorted_desserts)
    
    listOfKeys=list(sorted_desserts.values())
    print(listOfKeys)
    c=list(reversed(listOfKeys))
   

 


    return render_template("macros/macros.html",c=c,data=data)

# @app.route("/admin/profile")
# def admin_profile():
#     return "admin profile"

@app.route("/get-report/<path:path>")
def get_report(path):
    try:
        return send_from_directory(
            app.config["CLIENT_REPORTS"],filename=path ,as_attachmant=True
        )
    except FileNotFoundError:
       abort(404)
@app.route("/cookies")
def cookies():
    res = make_response("Cookies",200)

    cookies=request.cookies
    flavor=cookies.get("flavor")
    print("flavor")
    res.set_cookie(
        "flavor",
        value="chocolate chip", 
        max_age=10,
        
        expires=None,
        path=request.path,
        domain=None,
        secure=False,
        httponly=False,
        )


    return  res 
@app.route("/profile1")  
def profile1():
    
    if  session.get("USERNAME",None) is not None:
        username = session.get("USERNAME")
        
        user=users[username]
        return render_template("public/profile.html",user=user)
        
    else:
        
        return redirect(url_for("sign_in"))
@app.route("/sign-out")
def sign_out():
    session.pop("USERNAME")
    return redirect(url_for("sign_in"))

# stock = { "fruit":{
#           "apple":20,
#           "banana":45,
#            "cherry":1000
#           }}
# @app.route("/gest-text")
# def get_text():
#     return"some text"
# @app.route("/qs")
# def qs():
#     if request.args:
#         req=request.args
#         return" ".join(f"{k}:{v}" for k,v in req.items())
#     return "no query"

# @app.route("/stock")
# def get_stock():
#     res= make_response(jsonify(stock), 200)
#     return res

# @app.route("/stock/<collection>/<member>")
# def get_collection(collection,member):
#     if collection in stock:
#         member=stock[collection].get(member)
#         if member:
#             res=make_response(jsonify(member),200)
#             return res
    
#         res=make_response(jsonify({"error":"member Unknown"}),400)
#         return res
#     res=res=make_response(jsonify({"error":"collection not found"}),400)
#     return res

# @app.route("/stock/<collection>", methods=["DELETE"])
# def delete_collection(collection):
#     if collection in stock:
#        del stock[collection]
    
#        res=make_response(jsonify({}),204)
        
#        return res
#     res=make_response(jsonify({"error":"collection not found"}),400)
#     return res
# def count_words(url):
#     print(f"counting words at {url}")
#     start=time.time()
#     r= request.urlopen(url)
#     soup=BeautifulSoup(r.read().decode(), "lxml")
#     paragraphs=" ".join([p.text for p in soup.find.all("p")])
#     word_count=dict()
#     for i in paragraphs.split():
#         if not i in word_count:
#             word_count[i]=1
#         else:
#             word_count[i]+=1
#     end=time.time()
#     time_elapsed =end - start
#     print(word_count)
#     print(f"total words : {len(word_count)}")
#     print(f"total elapsed : {time_elapsed}")
#     return len(word_count)



    

    

        



#LIST OF HTTP  STATUS CODE/ HYPERTEXT TRANSFER PROTOCOM 
# get
# post
# # put
# # patch
# # delete
# r=redis.Redis()   
# q=Queue(connection=r)
# # def background_task(n):
# #     delay=2
# #     print("task running")
# #     print(f"simulating {delay}second delay")
# #     time.sleep(delay)
# #     print(len(n))
# #     print("task completed")
# #     return len(n)
# @app.route("/add_task", methods=["GET","POST"])
# def add_task():
#     # if request.args.get("n"):
#     #     job=q.enqueue(request.args.get("n"))
#     #     q_len=len(q)
#     #     return f"Task {job.id} added to queue at {job.enqueued_at},{q_len}tasks in the queue"
     
#      return "no value for n"

if __name__ == '__main__':
    app.run(debug=False)
