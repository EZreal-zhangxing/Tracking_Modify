import pickle
import numpy as np
from torch import tensor
import torch

import cv2

threshold = 0.5
frame_range = 3
iou_threshold = 0.4
def ioucalcul(box1,box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    if xmin1 >= xmax2 or xmax1 <= xmin2:
        return 0.
    if ymin1 >= ymax2 or ymax1 <= ymin2:
        return 0.
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    # 求交集部分右下角的点
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    # 计算总面积
    s = s1 + s2
    # 计算交集
    inter_area = (xmax - xmin) * (ymax - ymin)
    iou = inter_area / (s - inter_area)
    return iou
# 计算两帧之间的交幷比
def calFramesIou(targetFrame,otherFrame):
    target_boxs = targetFrame[3]
    other_boxs = otherFrame[3]
    iouresult = []
    for tbox in target_boxs:
        t = [tbox[0],tbox[1],tbox[0]+tbox[2],tbox[1]+tbox[3]]
        boxresult = []
        for obox in other_boxs:
            o = [obox[0],obox[1],obox[0]+obox[2],obox[1]+obox[3]]
            boxresult.append(ioucalcul(t,o))
        iouresult.append(boxresult)
    return np.array(iouresult)

def AnalyFrames(targetFrame,otherFrame):
    """
    返回目标框的个数，和满足阈值的个数
    :param targetFrame:
    :param otherFrame:
    :return:
    """
    if len(targetFrame[0]) == 0 or len(otherFrame[0]) == 0:
        return 0,0
    iouresult = calFramesIou(targetFrame, otherFrame)
    frames_result = np.max(iouresult, axis=1)  # 帧中框的最大的交幷比
    true_box = sum(frames_result > threshold)  # 帧中可信的框数
    return len(frames_result),true_box

def makeBox(all_position,score,rclass):
    all_need = []
    r_class = rclass[0][0]
    for index,position in enumerate(all_position[0],0):
        need_add = []
        scores = []
        box = [position[0],position[1],position[0]+position[2],position[1]+position[3]]
        box_score = score[0][index]
        scores.append(box_score)
        need_add.append(box)
        for i in range(1,len(all_position)):
            for index,o_po in enumerate(all_position[i],0):
                box2 = [o_po[0],o_po[1],o_po[0]+o_po[2],o_po[1]+o_po[3]]
                iou = ioucalcul(box,box2)
                if iou > iou_threshold:
                    need_add.append(box2)
                    scores.append(score[i][index])
        need_add = np.array(need_add)
        new_position = np.mean(need_add,axis = 0)
        scores = np.array(scores)
        mean_score = np.mean(scores)
        all_need.append({"mean_score" : mean_score,"position" : [new_position[0],new_position[1],new_position[2]-new_position[0],new_position[3]-new_position[1]],"class":r_class})
    return all_need

def analyseData(radars_list,frame_index):
    # 目标帧
    msg= []
    target_frame = radars_list[frame_index]

    if frame_index > 0 and frame_index < len(radars_list)-1:
        # 不是第一帧
        prev_frame = radars_list[frame_index-1]
        next_frame = radars_list[frame_index+1]

        # if len(prev_frame[3]) > len(target_frame[3]):
            # 前一帧的框数大于当前帧 判断当前帧是否应该 补框
        # 当前帧与后一帧进行交幷比计算
        _,true_box = AnalyFrames(prev_frame,next_frame)
        _,prev_true = AnalyFrames(target_frame,prev_frame)
        _,next_true = AnalyFrames(target_frame,next_frame)
        if true_box > prev_true and true_box > next_true:
            # 前后帧的 真框大于当前框 补框
            all_positions = []
            all_score = []
            all_rclass = []
            for num in range(len(radars_list)):
                position = []
                score = []
                rclass = []
                if num != frame_index:
                    o = radars_list[num]
                    boxs = calFramesIou(o,target_frame)
                    for row in range(boxs.shape[0]):
                        # 遍历每个框的交幷比大于阈值 该框存在于目标框
                        if sum(boxs[row] > threshold) > 0:
                            pass
                        else:
                            # 该框不存在与目标框
                            position.append(o[3][row])
                            score.append(o[0][row])
                            rclass.append(o[1][row])
                if len(position) > 0:
                    all_positions.append(position)
                    all_score.append(score)
                    all_rclass.append(rclass)
            allbox = makeBox(all_positions,all_score,all_rclass)
            for row in allbox:
                operation = {}
                operation["ope"] = "add"
                operation["score"] = row["mean_score"]
                operation["box"] = row["position"]
                operation["class"] = row["class"]
            msg.append(operation)
        elif true_box < prev_true and true_box < next_true:
            #删除框
            boxs = calFramesIou(target_frame,prev_frame)
            for row in range(boxs.shape[0]):
                operation = {}
                if sum(boxs[row] >threshold) > 0:
                    pass
                else:
                    operation["ope"] = "del"
                    operation["boxid"] = row
                    msg.append(operation)
        else:
            pass
        # else:
        #     # 前一帧的框数小于 当前框
        #
        #     pass
    else:
        # 是第一帧 或者最后一帧
        if frame_index == 0:
            pass
        elif frame_index == len(radars_list)-1:
            pass
    return msg
def drawImageAndBox(frame,key):
    frame_mat = cv2.imread("./city_5_0/radar_%d.jpg" % key)
    for box in frame[3]:
        x,y,w,h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(frame_mat,(x,y),(x+w,y+h),(255,0,0),2)
    frame_mat = cv2.resize(frame_mat, (500, 500))
    return frame_mat
def dramDelAndAddBox(addboxs,delarray,frame,key):
    frame_mat = cv2.imread("./city_5_0/radar_%d.jpg" % key)
    for i in range(len(frame[3])):
        x, y, w, h = frame[3][i]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if i not in delarray:
            cv2.rectangle(frame_mat, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame_mat, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for position in addboxs:
        x, y, w, h = position
        cv2.rectangle(frame_mat, (x, y), (x + w, y + h), (0, 0, 0), 2)
    frame_mat = cv2.resize(frame_mat,(500,500))
    return frame_mat
def drawAfterDealBox(frame,key):
    frame_mat = cv2.imread("./city_5_0/radar_%d.jpg" % key)
    for box in frame[3]:
        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(frame_mat, (x, y), (x + w, y + h), (200, 200, 200), 2)
    frame_mat = cv2.resize(frame_mat, (500, 500))
    return frame_mat
def groundTruth(key):
    frame_mat = cv2.imread("./citygroundtruth/radar_cart_vis_%d.png" % key)
    frame_mat = cv2.resize(frame_mat, (500, 500))
    return frame_mat
new_frame = {}
del_num = 0
add_num = 0
with open("./junction_radar_det_IN.pkl","rb") as file:
    f = dict(pickle.load(file))
    for key in f.keys():
        if key ==18:
            print("")
        print(key)
        print(f[key])
        frame_list = []
        frame_list_index = -1
        for i in range(-frame_range,frame_range+1):
            if i == 0 :
                frame_list_index = frame_list_index + 1
                frame_list.append(f[key])
            elif i < 0:
                if key + i  in new_frame.keys():
                    frame_list_index = frame_list_index + 1
                    frame_list.append(new_frame[key+i])
                    continue
                elif key + i in f.keys():
                    frame_list_index = frame_list_index + 1
                    frame_list.append(f[key + i])
                    continue
                else:
                    pass
            else:
                if key + i in f.keys():
                    frame_list.append(f[key+i])
        msg = analyseData(frame_list,frame_list_index)
        del_array = []
        add_box = []
        add_score = []
        add_class = []
        for message in msg:
            if message["ope"] == "del":
                del_array.append(message["boxid"])
            elif message["ope"] == "add":
                add_box.append(message["box"])
                add_score.append(message["score"])
                add_class.append(message["class"].item())
            else:
                raise Exception("operation type error!")
        new_position = []
        new_class = []
        new_score = []
        # 删除框
        del_num += len(del_array)
        if len(del_array) > 0:
            for i in range(len(f[key][3])):
                if i not in del_array:
                    new_position.append(f[key][3][i])
                    new_class.append(f[key][1][i].item())
                    new_score.append(f[key][0][i].item())
            new_class = tensor(new_class)
            new_score = tensor(new_score)
        else:
            new_position = f[key][3]
            new_class = f[key][1]
            new_score = f[key][0]
        add_num += len(add_box)
        if len(add_box) > 0:
            for boxs in add_box:
                new_position.append(boxs)
            add_class = tensor(add_class)
            add_score = tensor(add_score)
            new_class = torch.cat((new_class, add_class), dim=0)
            new_score = torch.cat((new_score, add_score), dim=0)
        new_frame[key] = [new_score,new_class,f[key][2],new_position,f[key][4],f[key][5]]
        # mat0 = groundTruth(key)
        # mat1 = drawImageAndBox(f[key],key)
        # mat2 = dramDelAndAddBox(add_box,del_array,f[key],key)
        # mat3 = drawAfterDealBox(new_frame[key],key)
        # imageradar = np.hstack((mat0,mat1,mat2,mat3))
        # cv2.imshow("truth-predict-deal-afterdeal",imageradar)
        # cv2.waitKey(0)
        # print(new_frame[key])
print(add_num,del_num)
with open("./radar_det_deal.pkl","wb") as wfile:
    pickle.dump(new_frame,wfile)