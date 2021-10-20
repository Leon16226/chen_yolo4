import numpy as np
from shapely.geometry import Polygon
import yaml
from .strategy import todo



# Tools-----------------------------------------------------------------------------------------------------------------
def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# nms
def func_nms(boxes, nms_threshold):
    b_x = boxes[:, 0]
    b_y = boxes[:, 1]
    b_w = boxes[:, 2] - boxes[:, 0]
    b_h = boxes[:, 3] - boxes[:, 1]

    areas = b_w * b_h

    scores = boxes[:, 5]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        i_other = order[1:]

        # inter area  : left_top   right_bottom
        xx1 = np.maximum(b_x[i], b_x[i_other])
        yy1 = np.maximum(b_y[i], b_y[i_other])
        xx2 = np.minimum(b_x[i] + b_w[i], b_x[i_other] + b_w[i_other])
        yy2 = np.minimum(b_y[i] + b_h[i], b_y[i_other] + b_h[i_other])
        # inter area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        # calc IoU
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        IoU = inter / union

        inds = np.where(IoU <= nms_threshold)[0]
        order = order[inds + 1]

    final_boxes = np.array([boxes[i] for i in keep])
    return final_boxes


# shapley
def Cal_area_2poly(point1, point2):
    poly1 = Polygon(point1).convex_hull
    poly2 = Polygon(point2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


# yaml
def readyaml():
    f = open('ydeepsort/config.yaml', 'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    return x

# track-----------------------------------------------------------------------------------------------------------------
def postprocess_track(outputs,
                      car_id_pool, people_id_pool, material_id_pool,
                      opt, im0s):

    # box : [x1, y1, x2, y2, id, cls]
    in_track_box = np.array(outputs)

    # 根据cls分流box
    c_box = {0: in_track_box[(in_track_box[:, 5] == 0) + (in_track_box[:, 5] == 1)],
             1: in_track_box[(in_track_box[:, 5] == 8) + (in_track_box[:, 5] == 0)],
             2: in_track_box[in_track_box[:, 5] == 2]}

    pool = [car_id_pool, people_id_pool, material_id_pool]
    todo(c_box, pool, opt, im0s)







# postprocess-----------------------------------------------------------------------------------------------------------
def postprocess(labels, infer_output, CLASS_SCORE_CONST,NMS_THRESHOLD_CONST,
                orig_shape, MODEL_HEIGHT, MODEL_WIDTH,
                point2, im0s, opt, threshold_box, point1, path, img, MODEL_PATH_EX, model_extractor):

    # init
    nc = len(labels)
    MODEL_OUTPUT_BOXNUM = infer_output.shape[1]

    # 1.process：-------------------------------------------------------------------------------------------------------
    result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
    list_class = infer_output[:, :, 5:5 + nc].reshape((-1, nc)).astype('float32')
    list_max = list_class.argmax(axis=1)
    list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
    result_box[:, 5] = list_max[:, 0]
    # conf
    list_max = list_class.max(axis=1)
    list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
    result_box[:, 4] = list_max[:, 0]

    # 2.整合
    boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)
    boxes[:, :4] = result_box[:, :4]
    boxes[:, 4] = result_box[:, 5]
    boxes[:, 5] = result_box[:, 4]
    all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]

    # filter
    # only people---------------------------------------------------------------------------------------------------
    # if (infer_output_size == 2):
    #     all_boxes = all_boxes[all_boxes[:, 4] == (8 or 2 or 3 or 4 or 5 or 6 or 7)]

    if all_boxes.shape[0] > 0:
        # 3.nms
        real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)


        # 4.scale
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        # filter strategy-----------------------------------------------------------------------------------------------

        # real_box------------------------------------------------------------------------------------------------------
        coords = []
        coords_in_area = []

        top_x = (real_box[:, 0] * 608 * x_scale).astype(int)
        top_y = (real_box[:, 1] * 608 * y_scale).astype(int)
        bottom_x = (real_box[:, 2] * 608 * x_scale).astype(int)
        bottom_y = (real_box[:, 3] * 608 * y_scale).astype(int)

        # if in detect area
        point = np.array([x for x in zip(top_x, top_y, top_x, bottom_y,
                                     bottom_x, bottom_y, bottom_x, top_y)]).reshape([-1, 4, 2])
        inter_area = np.array([Cal_area_2poly(point1, p) for p in point])
        # print("inter_area:", inter_area)
        in_area_box = real_box[inter_area > 5*5]

        # select strategy
        c_box = {"car": in_area_box[in_area_box[:, 4] == (0 or 1)],
                "people": in_area_box[in_area_box[:, 4] == 9],
                "material": in_area_box[in_area_box[:, 4] == (2 or 3 or 4 or 5 or 6 or 7)]}

        todo(c_box)










    # for detect_result in real_box:
    #
    #     top_x = int(detect_result[0] * 608 * x_scale)
    #     top_y = int(detect_result[1] * 608 * y_scale)
    #     bottom_x = int(detect_result[2] * 608 * x_scale)
    #     bottom_y = int(detect_result[3] * 608 * y_scale)
    #
    #     # plan1-----------------------------------------------------------------------------------------------------
    #     point = [top_x, top_y, top_x, bottom_y, bottom_x, bottom_y, bottom_x, top_y]
    #     point = np.array(point).reshape(4, 2)
    #     inter_area = Cal_area_2poly(point1, point)
    #     pred_area = (bottom_x - top_x) * (bottom_y - top_y)
    #     iou_p1 = inter_area / pred_area if pred_area != 0 else 0
    #
    #     # plan2-----------------------------------------------------------------------------------------------------
    #     if (iou_p1 >= 0.5):
    #         niou = 0
    #         for i, p in enumerate(point2[int(detect_result[4])]):
    #             p = np.array(p).reshape(4, 2)
    #             inter_area = Cal_area_2poly(point, p)
    #             p_iou = inter_area / pred_area
    #             niou = niou + 1 if p_iou >= 0.9 else niou
    #         print("niou:", niou)
    #
    #     # cv2
    #     threshold = 2
    #     if (iou_p1 >= 0.5 and niou < 5):
    #         coords_in_area.append((top_x, top_y, bottom_x - top_x, bottom_y - top_y,
    #                                detect_result[4], detect_result[5]))
    #     if (iou_p1 >= 0.5 and niou < threshold):
    #         cv2.rectangle(im0s, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 3)
    #         cv2.putText(im0s, labels[int(detect_result[4])] + " " + str(detect_result[5]), (top_x, top_y - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    #         coords.append((top_x, top_y, bottom_x - top_x, bottom_y - top_y, detect_result[4], detect_result[5]))
    #
    #
    #     # before push---------------------------------------------------------------------------------------------------
    #     if len(coords_in_area) > 0:
    #         # del-------------------------------
    #         for i, p2 in enumerate(point2):
    #             if (len(p2) >= threshold_box):
    #                 del p2[0:-threshold_box]
    #         # in---------------------------------
    #         for i, cor in enumerate(coords_in_area):
    #             point2[int(cor[4])].append([cor[0], cor[1], cor[0], cor[1] + cor[3],
    #                                     cor[0] + cor[2], cor[1] + cor[3], cor[0] + cor[2], cor[1]])
    #         # in--------------------------------
    #         if (len(coords) > 0):
    #             for i, cor in enumerate(coords):
    #                 for j in np.arange(0, 2):
    #                     point2[int(cor[4])].append([cor[0], cor[1], cor[0], cor[1] + cor[3],
    #                                             cor[0] + cor[2], cor[1] + cor[3], cor[0] + cor[2], cor[1]])
    #
    #
    #     # push----------------------------------------------------------------------------------------------------------
    #     if (len(coords) > 0):
    #         print("push one")
    #         push(opt, im0s, coords)






