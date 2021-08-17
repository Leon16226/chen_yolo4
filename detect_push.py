import json
import requests
import base64
import time
import datetime


def detect():

    # init time
    t0, t1 = 0., 0.

    # Initialize
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # Load model--------------------------------------------------------------------------------------------------------
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)

    # Set Dataloader----------------------------------------------------------------------------------------------------
    vid_path, vid_writer = None, None
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # iter
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        orig_shape = im0s.shape[:2]
        resized_img = img.astype(np.float32)
        resized_img /= 255.0

        # 模型推理
        t = time.time()
        infer_output = model.execute(resized_img)
        infer_output = infer_output[0]
        t0 += time.time() - t

        # 1.根据模型的输出以及对检测网络的认知，可以知道：-------------------------------------------------
        result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
        list_class = infer_output[:, :, 5:7].reshape((-1, 2)).astype('float32')
        list_max = list_class.argmax(axis=1)
        list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]

        # 2.整合
        boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)  # 创建一个
        boxes[:, :4] = result_box[:, :4]
        boxes[:, 4] = result_box[:, 5]
        boxes[:, 5] = result_box[:, 4]
        all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]

        # 3.nms
        t = time.time()
        real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)
        t1 += time.time() - t

        # 4.scale
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        # im0s -> h, w, n
        for detect_result in real_box:
            top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)
            cv2.rectangle(im0s, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1)
            cv2.putText(im0s, labels[int(detect_result[4])], (top_x, top_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print('im0s', im0s.shape)

        # save----------------------------------------------------------------------------------------------------------
        save_path = str(Path(out) / 'result.mp4')
        fourcc = 'mp4v'
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(fourcc, fps, w, h)
        if i == 0:
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))  # 2560 * 1536
        # vid_writer.write(im0s)

        cv2.imshow('mask', im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
            break

    vid_writer.release()

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
        img = img[2:]

    event = Event('10.17.1.20', int(round(time.time() * 1000)),
                  1, "yzw1-dxcd", "illegalPark", "", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  "people", 0, 0, 0, 0, 0.75,
                  "", "",
                  "")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)

    # post -------------------------------------------------------------------------------------------------------------
    url = 'http://192.168.1.19:8080/v1/app/interface/uploadEvent'
    headers = {"content-type": "application/json"}

    ret = requests.post(url, data=event, headers=headers)

    print(ret.text)

    # parse json
    # json_str = '{'age': 29, 'name': 'tom'}'
    # def handle(d):
    #       return Man(d['name], d['age')
    # m = json.loads(json_str, object_hook=handle)


if __name__ == '__main__':

    push()
