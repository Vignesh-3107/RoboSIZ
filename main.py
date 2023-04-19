import os
import sys
from flask import Flask,request, render_template, abort, url_for, json, jsonify
import json
import time 

import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt
from threading import Thread
import webcamera
from lanyocr import LanyOcr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
    
Coords = []
Centroid = []
Scanned = []
Scanned.append(0)
Scanned.append(0)
x_real = 30
x_unitd = 21.3333333333
y_real = 17
y_unitd = 28.2352941176

with open('./templates/ObjectDetectLog.json',"w") as myfile:
    JsStart = {
        "@Description": {
        "_Class": "Class of Object",
        "_Accuracy": "Accuracy level of Detection",
        "_Time": "Date and Time of Execution"
    }
    }
    JsStart = json.dumps(JsStart)
    myfile.write(JsStart)
with open('./templates/OCRLog.json',"w") as myfile:
    JsStart = {
        "@Description": {
        "_Text": "Detected Text",
        "_CER": "Character Error Rate",
        "_Time": "Date and Time of Execution"
    }
    }
    JsStart = json.dumps(JsStart)
    myfile.write(JsStart)

def findCoordinates(position):
    for i in range(0,4):
        coordinates = torch.tensor(position[i]).item()
        Coords.append(coordinates)
    # Centroid.append((Coords[0] + Coords[2])/2)
    # Centroid.append((Coords[1] + Coords[3])/2)
    Dic = {
        "X":round(((Coords[0] + Coords[2])/2)/x_unitd, 3),
        "Y":round(((Coords[1] + Coords[3])/2)/y_unitd, 3),
        "Z": 28
    }
    json_obj = json.dumps(Dic)
    with open("Input.json", "w") as outfile:
        outfile.write(json_obj)
    print("******* Finished *********")

def CER(Actual,Detected):
        total = len(Actual)
        itera = len(Detected)
        matched = 0
        if total >= itera:
            length = itera
        else:
            length = total
        for i in range(length):
            if Actual[i] == Detected[i]:
                matched = matched + 1
        return str(((total - matched)*100)/total)



def Ocrmain(image_path, 
         output_path=" ",
        detector_name ="paddleocr_en_ppocr_v3",
        recognizer_name ="paddleocr_en_mobile",
        merger_name ="lanyocr_nomerger",
        merge_rotated_boxes =True,
        merge_vertical_boxes =True,
        merge_boxes_inference =False,
        use_gpu =False,
        debug=False,
        ):
    ocr = LanyOcr(
        detector_name=detector_name,
        recognizer_name=recognizer_name,
        merger_name=merger_name,
        merge_boxes_inference=merge_boxes_inference,
        merge_rotated_boxes=merge_rotated_boxes,
        merge_vertical_boxes=merge_vertical_boxes,
        use_gpu=use_gpu,
        debug=debug,
    )

    results = ocr.infer_from_file(image_path)
    # print(results)
    return results
     
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        userinput=' ' # user input
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
     

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                readImg = imc.copy()
                LabelName = ["Component","Text"]
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    local = time.ctime().split(" ")
                    timeNow = local[2]+"-"+local[1]+"-"+local[4]+" "+local[3]
                    with open("./templates/ObjectDetectLog.json","r") as OBfile:
                        ODfile = json.load(OBfile)
                    OBnew ={}
                    OBdatum ={}
                    OBdatum["_Class"] = LabelName[int(cls)]
                    OBdatum["_Accuracy"] = str(round(torch.tensor(conf).item() * 100, 3))
                    OBdatum["_Time"] = timeNow
                    OBnew[str(len(ODfile))] = OBdatum
                    ODfile.update(OBnew)
                    with open("./templates/ObjectDetectLog.json", "w") as refile:
                        json.dump(ODfile, refile,indent=4)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # print(label)
                        if names[c]=="Component" or label.split(" ")[0] == "Component":
                            Scanned[0] = Scanned[0] + 1
                        if names[c]=="Text" or label.split(" ")[0] == "Text":
                            Scanned[1] = Scanned[1] + 1
                            output = Ocrmain(image_path=readImg)
                            # with open('./static/UserIn.txt',"r") as txtfile:
                            #     userinput = txtfile.read()
                            #     txtfile.close()
                            if output != []:
                                with open("./templates/OCRLog.json","r") as file:
                                    file = json.load(file)
                                for item in output:
                                    new ={}
                                    datum ={}
                                    mouldTextData = item.text
                                    cer = CER(userinput.lower(),mouldTextData.lower())
                                    datum["_Text"] = mouldTextData
                                    datum["_CER"] = cer
                                    datum["_Time"] = timeNow
                                    new[str(len(file))] = datum
                                    file.update(new)
                                    with open("./templates/OCRLog.json", "w") as refile:
                                        json.dump(file, refile,indent=4)
                                    if mouldTextData.lower() == userinput.lower():
                                        findCoordinates(xyxy)
                                        LoadStreams.exitCam()
                                        return 1
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
          

            # Stream results
            # im0 = annotator.result()
            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 print(fps)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # print(s)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--userinput', type=str, default=None, help='User input to search')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def OCR_parse_opt():
    parser = argparse.ArgumentParser(description="LanyOCR")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="./outputs/output1.jpg")
    parser.add_argument(
        "--detector_name",
        type=str,
        default="paddleocr_en_ppocr_v3",
        help='Name of detector, must be one of these ["easyocr_craft", "paddleocr_en_ppocr_v3"]',
    )
    parser.add_argument(
        "--recognizer_name",
        type=str,
        default="paddleocr_en_mobile",
        help='Name of recognizer, must be one of these ["paddleocr_en_server", "paddleocr_en_mobile", "paddleocr_en_ppocr_v3", "paddleocr_french_mobile", "paddleocr_latin_mobile"]',
    )
    parser.add_argument(
        "--merger_name",
        type=str,
        default="lanyocr_nomerger",
        help="Name of merger, must be one of these ['lanyocr_nomerger', 'lanyocr_craftbased']",
    )
    parser.add_argument(
        "--merge_rotated_boxes",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="enable merging text boxes in upward/downward directions",
    )
    parser.add_argument(
        "--merge_vertical_boxes",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="enable merging text boxes in vertical direction",
    )
    parser.add_argument(
        "--merge_boxes_inference",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="merge boxes in a text line into images before running inferece to speedup recognition step",
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="use GPU for inference",
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="generate text lines and text boxes images for debugging purposes",
    )
    args = parser.parse_args()
    return args

   
app = Flask(__name__)

with open('./templates/FullAccess.json', 'r') as myfile:
    data = myfile.read()

Products = []
@app.route("/")
def index():
    return render_template('index.html',title="page", DataBase=json.dumps(data))

@app.route("/home")
def home():
    return render_template('index.html',title="page", DataBase=json.dumps(data))

@app.route("/result",methods=["POST"])
def result():
    product = request.form["ProductId"]
    print(product)
    opt = parse_opt()
    opt.weights = ["last.pt"]
    opt.source = "1"
    opt.userinput = product
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    with open('./Input.json', 'r') as myfile:
        Cord = myfile.read()
    return render_template('result.html', DataCord=json.dumps(Cord), ExeCount=Scanned)

@app.route("/ocroutput")
def ocroutput():
    with open('./templates/OCRLog.json', 'r') as myfile:
        OcrLog = myfile.read()
    return render_template('ocroutput.html',title="page", OCRData=json.dumps(OcrLog))

@app.route("/objectdetection")
def objectdetection():
    with open('./templates/ObjectDetectLog.json', 'r') as myfile:
        DetectLog = myfile.read()
    return render_template('objectdetection.html',title="page", ODData=json.dumps(DetectLog))

@app.route("/thank")
def thank():
    return render_template('thank.html',title="page")

@app.route("/exitout")
def exitout():
    sys.exit(0)

if __name__ == "__main__":
    app.run(debug=True)
