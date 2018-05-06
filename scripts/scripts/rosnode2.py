#!/usr/bin/env python
import rospy
import cv2
import roslib
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#model = ResNet50(weights='imagenet')
#model=VGG16(weights='imagenet')



model_path = '/home/esraa/catkin_ws/src/roomba_robot/scripts/scripts/models/model.h5'
model_weights_path = '/home/esraa/catkin_ws/src/roomba_robot/scripts/scripts/models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
model._make_predict_function()
graph = tf.get_default_graph()
target_size = (224, 224)


rospy.init_node('classify', anonymous=True)

pub = rospy.Publisher('object_detected', String, queue_size = 1)
pub1 = rospy.Publisher('object_detected_probability', Float32, queue_size = 1)
bridge = CvBridge()

msg_string = String()
msg_float = Float32()
def change(lable_num):
    lable_num=lable_num.astype(float)  
    
    if lable_num == 0:
       print("Label: fridge")
    elif lable_num == 1:
       print("Label: stove")
    elif lable_num == 2:
       print("Label: eattable")
    elif lable_num == 3:
       print("Label: sink")
 
    elif lable_num == 4:
       print("Label: kitcheb_locker ")
    elif lable_num == 5:
       print("Label:floor ")
    return lable_num 





        




def callback(image_msg):
  
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_image = cv2.resize(cv_image, target_size) 
    #np_image = np.asarray(cv_image)  
    np_image=img_to_array(cv_image)            
    np_image = np.expand_dims(np_image, axis=0)   
    #np_image = np_image.astype(float)  
    #np_image = preprocess_input(np_image)        
    
    global graph                                   
    with graph.as_default():
       preds = model.predict(np_image) 
       

       
       result = preds[0]
       
       answer = np.argmax(result)
       print ("answer ",change(answer))
       
      
       msg_string.data = answer
       msg_float.data = float(answer)
       pub.publish(msg_string)
       pub1.publish(msg_float)      

rospy.Subscriber("/camera/rgb/image_raw", Image, callback, queue_size = 1, buff_size = 16777216)



while not rospy.is_shutdown():
  rospy.spin()
