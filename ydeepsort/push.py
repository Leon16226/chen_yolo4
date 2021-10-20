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


def push(opt, frame, events):
    print("post a event:" + events)

    # opt
    post_url = opt.post
    ponit_ip = opt.point

    # event ------------------------------------------------------------------------------------------------------------
    _, bi_frame = cv2.imencode('.jpg', frame)
    img = base64.b64encode(bi_frame)
    img = str(img)
    img = img[2:]

    # UTC -> CST
    # now_date = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
    # now_stamp = int(time.mktime(time.strptime(now_date, '%Y-%m-%d %H:%M:%S'))) * 1000   # 毫秒级时间戳



    event = Event(ponit_ip, int(round(time.time() * 1000)),
                  0, "路段2", events, "",
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  "People", 0, 0, 0, 0, 0.75,
                  "", "",
                  "1")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)


    # post -------------------------------------------------------------------------------------------------------------
    url = post_url
    headers = {"content-type": "application/json"}
    ret = requests.post(url, data=event, headers=headers)
    print("post result:")
    print(ret.text + "###################################################")
