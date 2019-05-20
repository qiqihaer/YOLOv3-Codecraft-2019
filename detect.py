# import argparse
import time
# from sys import platform

from models import *
from datasets import *


def detect():

    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5

    device = torch_utils.select_device()
    output = 'output'
    save_txt = False,
    save_images = True,
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model

    cfg = 'yolov3-spp.cfg'
    model = Darknet(cfg, img_size)

    PATH = './model/model.pth'
    model.load_state_dict(torch.load(PATH))




    model.fuse()
    model.to(device).eval()


    images = './data/data'
    dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '深', '秦', '京', '海', '成',
               '南', '杭', '苏', '松']


    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        # input_batch = []
        # input_batch.append(img)
        # input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)



        pred = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            # det1 = det
            det[:, :4] = scale_coords((96, 416), det[:, :4], (70, 360, 3)).round()
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            if det.shape[0] > 9:
                det = det[0:9, :]
            _, sort_idx = torch.sort(det[:, 0])
            det1 = det[sort_idx]

            l_char = det1[:, 6].cpu().numpy()
            c_result = ''
            for j in range(len(l_char)):
                c_result += classes[int(l_char[j])]

            print(i, ':', c_result)


            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        cv2.imwrite(save_path, im0)



if __name__ == '__main__':

    with torch.no_grad():
        detect()
