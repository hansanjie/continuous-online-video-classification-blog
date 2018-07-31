"""
Run the online classification system.

Capture an image, classify, do it again.
"""
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import tensorflow as tf

from pythonosc import osc_message_builder
from pythonosc import udp_client

import cv2
import numpy as np


def get_labels():
    """Get a list of labels so we can see if it's an ad or not."""
    with open('retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
        #print(labels)
    return labels

def run_classification(labels):
    """Stream images off the camera and process them."""
    
    sendToAddress = "192.168.0.20"
    sendToPort = 7000
    oscName = "/AAAA"
    
    client = udp_client.SimpleUDPClient(sendToAddress,sendToPort)
    
    
    
    
    
    camera = PiCamera()
    camera.resolution = (1280, 720)
    camera.framerate = 10
    raw_capture = PiRGBArray(camera, size=(1280, 720))

    # Warmup...
    time.sleep(2)

    # Unpersists graph from file
    with tf.gfile.FastGFile('retrained_graph.pb', 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        
        # And capture continuously forever.
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for _, image in enumerate(
                camera.capture_continuous(
                    raw_capture, format='bgr', use_video_port=True
                )
            ):
            # Get the numpy version of the image.
            #image2 = cv2.resize(image, (224, 224))
            #image2 = np.asarray(cv2.GetMat(image))
            image2  = cv2.resize(image,new_shape=[224, 224, 3])
            decoded_image = image2.array.reshape(1, 224, 224, 3)

            cv2.imshow("Frame",image.array)
            key = cv2.waitKey(1) & 0xFF
            # Make the prediction. Big thanks to this SO answer:
            # http://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
            predictions = sess.run(softmax_tensor, {'Placeholder:0': decoded_image})
            prediction = predictions[0]

            # Get the highest confidence category.
            prediction = prediction.tolist()
            max_value = max(prediction)
            max_index = prediction.index(max_value)
            predicted_label = labels[max_index]

            print("%s (%.2f%%)" % (predicted_label, max_value * 100))
            
            message = [max_index,predicted_label,max_value]
            client.send_message(oscName,message)
            # Reset the buffer so we're ready for the next one.
            raw_capture.truncate(0)
            if key ==ord("q"):
                break

if __name__ == '__main__':
    run_classification(get_labels())
