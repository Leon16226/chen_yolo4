import cv2
import base64
import time
import datetime
import json
import requests

# Event  Post-----------------------------------------------------------------------------------------------------------
class Event(object):
    def __init__(self, cameraIp, timestamp,
                 roadId, roadName, code, subCode, dateTime, status, no, distance, picture,
                 targetType, xAxis, yAxis, height, width, prob,
                 miniPicture, carNo,
                 remark
                 ):
        self.cameraIp = cameraIp
        self.timestamp = timestamp
        self.events = [
            {
                "roadId": roadId,
                "roadName": roadName,
                "code": code,
                "subCode": subCode,
                "dateTime": dateTime,
                "status": status,
                "no": no,
                "distance": distance,
                "picture": picture,
                "coordinate": [
                    {
                        "targetType": targetType,
                        "xAxis": xAxis,
                        "yAxis": yAxis,
                        "height": height,
                        "width": width,
                        "prob": prob
                    }
                ],
                "carNoAI": {
                    "miniPicture": miniPicture,
                    "carNo": carNo
                },
                "remark": remark
            }
        ]


class Coordinate(object):
    def __init__(self, targetType, xAxis, yAxis, height, width, prob):
        self.targetType = targetType
        self.xAxis = xAxis
        self.yAxis = yAxis
        self.height = height
        self.width = width
        self.prob = prob


def push(opt, frame, coords):
    # opt
    post_url = opt.post
    ponit_ip = opt.point

    # event ------------------------------------------------------------------------------------------------------------
    _, bi_frame = cv2.imencode('.jpg', frame)
    img = base64.b64encode(bi_frame)
    img = str(img)
    img = img[2:]

    coordinate = []
    for i, coord in enumerate(coords):
        coordinate.append(Coordinate("material", coord[0], coord[1], coord[2], coord[3], coord[4]))

    event = Event(ponit_ip, int(round(time.time() * 1000)),
                  1, "yzw1-dxcd", "peopleOrNoVehicles" if opt.om == "weights/highway-sim.om" else "throwThings", "",
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  "material", 0, 0, 0, 0, 0.75,
                  "", "",
                  "")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)

    # post -------------------------------------------------------------------------------------------------------------
    url = post_url
    headers = {"content-type": "application/json"}
    ret = requests.post(url, data=event, headers=headers)
    print(ret.text)
