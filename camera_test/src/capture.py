import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam
from tensorflow.keras.models import load_model
import numpy as np



if __name__ == "__main__":

    # folder to write images to
    #out_folder = sys.argv[1]
    #["book","coin","cup","cutlery","face","gesture","glass","nail_polish","pen","plant","plate","shoe"]
    dic={0:"book",1:"coin",2:"cup",3:"cutlery",4:"face",5:"gesture",6:"glass",7:"nail_polish",8:"pen", 9:"plant", 10:"plate", 11:"shoe"}
    #load model
    network = load_model("models/model_resnet.h5")
    
    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                image_reshaped=image.reshape(1,224,224,3)
                item_class=np.argmax(network.predict(image_reshaped))
                print("The foto is:     "+dic[item_class])
                
                #write_image(out_folder, image) 

            # disable ugly toolbar  
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
