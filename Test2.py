import pickle
import numpy as np
import cv2
import json
rfile = open("./radar_det_0_5.pkl","rb")
deal_file = open("./radar_det_deal.pkl","rb")
info = dict(pickle.load(rfile))
deal_info = dict(pickle.load(deal_file))
for key in info.keys():
    radar = cv2.imread("./city_5_0/radar_{}.jpg".format(key))
    radar_deal = cv2.imread("./city_5_0/radar_{}.jpg".format(key))
    radars = info[key][3]
    deal_radar = deal_info[key][3]
    for position in radars:
        x,y = int(position[0]),int(position[1])
        width,heigth = int(position[2]),int(position[3])
        cv2.rectangle(radar, (x, y), (x + width, y + heigth), (0, 0, 255), 4)
    for position in deal_radar:
        x, y = int(position[0]), int(position[1])
        width, heigth = int(position[2]), int(position[3])
        cv2.rectangle(radar_deal, (x, y), (x + width, y + heigth), (0, 255, 0), 4)
    groundtruth = cv2.imread("./citygroundtruth/radar_cart_vis_%d.png" % key)
    new_radar = np.hstack((groundtruth,radar,radar_deal))
    new_radar = cv2.resize(new_radar,(1500,500))
    cv2.imshow("radar",new_radar)
    cv2.waitKey(0)

