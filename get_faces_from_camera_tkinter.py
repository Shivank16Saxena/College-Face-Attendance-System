import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk

detector = dlib.get_frontal_face_detector()

class FaceRegisterApp:
    def __init__(self):

        self.current_frame_faces_cnt = 0  
        self.existing_faces_cnt = 0  
        self.ss_cnt = 0  

        self.win = tk.Tk()
        self.win.title("Face Register")

        self.frame_top = tk.Frame(self.win)
        self.frame_top.pack(side=tk.TOP)

        self.font_info = tkFont.Font(family='Arial', size=14)
        self.label_fps_info = tk.Label(self.frame_top, text="", font=self.font_info)
        self.label_fps_info.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

        self.label_face_cnt = tk.Label(self.frame_top, text="Faces in current frame: ", font=self.font_info)
        self.label_face_cnt.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

        self.label_warning = tk.Label(self.frame_top, text="", font=self.font_info, fg='red')
        self.label_warning.grid(row=2, column=0, columnspan=2, padx=20, pady=10)

        self.log_all = tk.Label(self.frame_top, font=self.font_info)
        self.log_all.grid(row=3, column=0, columnspan=2, padx=20, pady=10)

        self.frame_registration = tk.Frame(self.frame_top)
        self.frame_registration.grid(row=4, column=0, padx=20, pady=10)

        self.font_reg = tkFont.Font(family='Arial', size=16, weight='bold')
        self.label_name = tk.Label(self.frame_registration, text="Name: ", font=self.font_reg)
        self.label_name.grid(row=0, column=0, padx=5, pady=10)

        self.input_name = tk.Entry(self.frame_registration, font=self.font_reg)
        self.input_name.grid(row=0, column=1, padx=5, pady=10)

        self.btn_input = tk.Button(self.frame_registration, text='Input', command=self.get_input_name, font=self.font_reg)
        self.btn_input.grid(row=0, column=2, padx=5, pady=10)

        self.btn_clear = tk.Button(self.frame_registration, text='Clear', command=self.clear_data, font=self.font_reg)
        self.btn_clear.grid(row=0, column=3, padx=5, pady=10)

        self.btn_save = tk.Button(self.frame_registration, text='Save Current Face', command=self.save_current_face, font=self.font_reg)
        self.btn_save.grid(row=0, column=4, padx=5, pady=10)

        self.label_database = tk.Label(self.frame_registration, text="Faces in database: ", font=self.font_reg)
        self.label_database.grid(row=0, column=5, padx=5, pady=10)

        self.label_cnt_face_in_database = tk.Label(self.frame_registration, text=str(self.existing_faces_cnt), font=self.font_reg)
        self.label_cnt_face_in_database.grid(row=0, column=6, padx=5, pady=10)


        self.frame_bottom = tk.Frame(self.win)
        self.frame_bottom.pack(side=tk.BOTTOM)

        self.label = tk.Label(self.frame_bottom)
        self.label.pack()

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0) 

    def clear_data(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images removed!"

    def get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def pre_work_mkdir(self):
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)
        else:
            self.existing_faces_cnt = 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt) + "_" + \
                                    self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)
        self.ss_cnt = 0
        self.face_folder_created_flag = True

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not go out of range!"
            else:
                self.log_all["text"] = "No face in the current frame!"
        else:
            self.log_all["text"] = "Please give name of the person"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)

        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    app = FaceRegisterApp()
    app.run()

if __name__ == '__main__':
    main()

