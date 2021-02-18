#Third Party
import tensorflow.compat.v1 as tf  # for TF 2
import tensornets as nets
import cv2
import numpy as np
import base64
tf.disable_v2_behavior() 


#Global Variables

#RTSP feed test: "rtsp://admin:admin0000@192.168.1.7:554/cam/realmonitor?channel=2&subtype=1"
class InferenceEngine:
    
    classes =  {'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
    def __init__(self, camara_feed = 0):
        self.classes = {'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
        self.camera_feed = camera_feed
    def load(self):
        
        inputs = tf.placeholder(tf.float32, [None, 416, 416, 3]) 
        model = nets.YOLOv3COCO(inputs, nets.Darknet19)
        sess = tf.Session()
        sess.run(model.pretrained())
        try:
            print("loading")
            cap = cv2.VideoCapture(self.camera_feed)
            print("connected")
        except:
            cap = np.zeros((416, 416))
            print("error")
            return inputs, model, sess, cap, True
        
        return inputs, model, sess, cap, False
    def inference(self, inputs, model, sess, cap,  classes = classes):
        if cap.isOpened():
            ret, frame = cap.read()
            img=cv2.resize(frame,(416,416))
            imge=np.array(img).reshape(-1,416,416,3)
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
            boxes = model.get_boxes(preds, imge.shape[1:3])
            image = self.packageImage(img)
            package = {}
            for i in classes:
                package[classes[i]] = boxes[int(i)].tolist()
        return package, boxes, image
    def packageImage(self, img):
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)





