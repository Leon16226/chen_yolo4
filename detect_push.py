import json
import requests
import base64
import time
import datetime


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

def push():
    # event ------------------------------------------------------------------------------------------------------------
    with open("/home/chen/chen_p/chen_yolo4/inference/output/mss1.jpg", "rb") as f:
        img = base64.b64encode(f.read())
        img = str(img)


    event = Event('10.17.1.20', int(round(time.time() * 1000)),
                  1, "yzw1-dxcd", "illegalPark", "", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  "people", 0, 0, 0, 0, 0.75,
                  "", "",
                  "")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)

    # post -------------------------------------------------------------------------------------------------------------
    url = 'http://192.168.1.19:8080/v1/app/interface/uploadEvent'
    payload = event
    headers = {"content-type": "application/json"}

    ret = requests.post(url, data=payload, headers=headers)

    print(ret.text)

    # parse json
    # json_str = '{'age': 29, 'name': 'tom'}'
    # def handle(d):
    #       return Man(d['name], d['age')
    # m = json.loads(json_str, object_hook=handle)


if __name__ == '__main__':

    push()
