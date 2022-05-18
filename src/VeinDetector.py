import cv2
from ExtractFeaturs import VeinDetection
from Preprocessing import adjust_gamma,creatMask
import tkinter as tk

def start():
    vid = cv2.VideoCapture(1)
    vid.set(10, 150)
    while(True):
        ret, image = vid.read()
        image=cv2.resize(image,(600,600))
        image=creatMask(image)
        image=adjust_gamma(image,2)
        #cv2.imshow('frame', image)
        grayImg= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_t = VeinDetection(image,grayImg)

        cv2.imshow('frame',img_t)
        if cv2.waitKey(32) == ord(' '):
            break

    cv2.destroyAllWindows()
    vid.release()
    root.destroy()



if __name__ == '__main__' :
    root = tk.Tk()
    canvas1 = tk.Canvas(root, width=300, height=300)
    canvas1.pack()

    openBtn = tk.Button(text='Open Camera', command=start, bg='brown', fg='white')
    label1 = tk.Label(root, text='press space to close camera', fg='green', font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 200, window=label1)
    canvas1.create_window(150, 150, window=openBtn)

    root.mainloop()
