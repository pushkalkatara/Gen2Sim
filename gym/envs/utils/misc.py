import numpy as np
import yaml
import os

def _draw_line(task, src, dst, clear = True, env_id = 0, color=np.array([1.0, 0.0, 0.0]).astype(np.float32)):
    line_vec = np.stack([src, dst]).flatten().astype(np.float32)
    if clear:
        task.gym.clear_lines(task.viewer)
    # print(env_id)
    task.gym.add_lines(
        task.viewer,
        task.env_ptr_list[env_id],
        task.env_num,
        line_vec,
        color
    )
    
def _draw_cross(task, dst, clear = True, env_id = 0, color=np.array([1.0, 0.0, 0.0]).astype(np.float32)):
    
    _draw_line(task, dst - np.array([0, 0, 0.5]), dst + np.array([0, 0, 0.5]), clear=clear, env_id=0, color=color) 
    _draw_line(task, dst - np.array([0, 0.5, 0]), dst + np.array([0, 0.5, 0]), clear=False, env_id=0, color=color) 
    _draw_line(task, dst - np.array([0.5, 0, 0]), dst + np.array([0.5, 0, 0]), clear=False, env_id=0, color=color) 

def draw_bbox(img, bbox_list, K = None):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        # bbox = bbox * trans[0]+trans[1:4]
        if K == None:
            K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        point2image = []
        for pts in bbox:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
            y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
            point2image.append([x_new, y_new])
        cl = [255,0,0]
        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[1],color=(255,0,0),thickness=1)
        cv2.line(img,point2image[1],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[2],color=(0,255,0),thickness=1)
        cv2.line(img,point2image[2],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[3],color=(0,0,255),thickness=1)
        cv2.line(img,point2image[3],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
    return img