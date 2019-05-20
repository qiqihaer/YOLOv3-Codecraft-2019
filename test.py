import argparse

from torch.utils.data import DataLoader

from models import *
from datasets import *
from utils import *


def test(
        val_files,
        cfg,
        weights=None,
        img_size=416,
        model=None
):
    if model is None:
        device = torch_utils.select_device()

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device

    # Configure run

    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '深', '秦', '京', '海', '成',
             '南', '杭', '苏', '松']

    dataloader = LoadImages(val_files, img_size=img_size)

    model.eval()

    conf_thres = 0.5
    nms_thres = 0.5

    acc = 0
    tot = 0

    for i, (img, labels) in enumerate(dataloader):

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            # det1 = det
            det[:, :4] = scale_coords((96, 416), det[:, :4], (70, 360, 3)).round()

            _, sort_idx = torch.sort(det[:, 0])
            det = det[sort_idx]

            det_numpy = det.cpu().detach().numpy()

            if det_numpy.shape[0] > 9:
                l = det_numpy.shape[0] - 9
                for k in range(l):
                    d = det_numpy[:, 0]
                    dc = np.zeros(len(d) - 1)
                    for j in range(len(dc)):
                        dc[j] = d[j + 1] - d[j]
                    row1 = np.argsort(dc)[0]
                    row2 = row1 + 1
                    conf1 = det_numpy[row1, 4]
                    conf2 = det_numpy[row2, 4]

                    if conf1 > conf2:
                        det_numpy = np.delete(det_numpy, row2, 0)
                    else:
                        det_numpy = np.delete(det_numpy, row1, 0)

            l_char = det_numpy[:, 6]

            c_result = ''
            for j in range(len(l_char)):
                c_result += names[int(l_char[j])]

            for k in range(min(len(c_result), len(labels))):
                if labels[k] == c_result[k]:
                    acc = acc + 1

            tot = tot + 9

    return acc/tot

















    # # coco91class = coco80_to_coco91_class()
    # # print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    # # loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    # # jdict, stats, ap, ap_class = [], [], [], []
    # # for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
    # for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
    #
    #     targets = targets.to(device)
    #     imgs = imgs.to(device)
    #
    #     # Plot images with bounding boxes
    #     if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
    #         plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')
    #
    #     # Run model
    #     inf_out, train_out = model(imgs)  # inference and training outputs
    #
    #     # Compute loss
    #     if hasattr(model, 'hyp'):  # if model has loss hyperparameters
    #         loss_i, _ = compute_loss(train_out, targets, model)
    #         loss += loss_i.item()
    #
    #     # Run NMS
    #     output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
    #
    #     # Statistics per image
    #     for si, pred in enumerate(output):
    #         labels = targets[targets[:, 0] == si, 1:]
    #         nl = len(labels)
    #         tcls = labels[:, 0].tolist() if nl else []  # target class
    #         seen += 1
    #
    #         if pred is None:
    #             if nl:
    #                 stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
    #             continue
    #
    #         # Append to pycocotools JSON dictionary
    #         # if save_json:
    #         #     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
    #         #     image_id = int(Path(paths[si]).stem.split('_')[-1])
    #         #     box = pred[:, :4].clone()  # xyxy
    #         #     scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
    #         #     box = xyxy2xywh(box)  # xywh
    #         #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    #         #     for di, d in enumerate(pred):
    #         #         jdict.append({
    #         #             'image_id': image_id,
    #         #             'category_id': coco91class[int(d[6])],
    #         #             'bbox': [float3(x) for x in box[di]],
    #         #             'score': float(d[4])
    #         #         })
    #
    #         # Assign all predictions as incorrect
    #         correct = [0] * len(pred)
    #         if nl:
    #             detected = []
    #             tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
    #
    #             # Search for correct predictions
    #             for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
    #
    #                 # Break if all targets already located in image
    #                 if len(detected) == nl:
    #                     break
    #
    #                 # Continue if predicted class not among image classes
    #                 if pcls.item() not in tcls:
    #                     continue
    #
    #                 # Best iou, index between pred and targets
    #                 iou, bi = bbox_iou(pbox, tbox).max(0)
    #
    #                 # If iou > threshold and class is correct mark as correct
    #                 if iou > iou_thres and bi not in detected:  # and pcls == tcls[bi]:
    #                     correct[i] = 1
    #                     detected.append(bi)
    #
    #         # Append statistics (correct, conf, pcls, tcls)
    #         stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    #
    # # Compute statistics
    # stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    # if len(stats):
    #     p, r, ap, f1, ap_class = ap_per_class(*stats)
    #     mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    #
    # # Print results
    # pf = '%20s' + '%10.3g' * 6  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')
    #
    # # Print results per class
    # if nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
    #
    # # # Save JSON
    # # if save_json and map and len(jdict):
    # #     imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
    # #     with open('results.json', 'w') as file:
    # #         json.dump(jdict, file)
    # #
    # #     from pycocotools.coco import COCO
    # #     from pycocotools.cocoeval import COCOeval
    # #
    # #     # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    # #     cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
    # #     cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api
    # #
    # #     cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # #     cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    # #     cocoEval.evaluate()
    # #     cocoEval.accumulate()
    # #     cocoEval.summarize()
    # #     map = cocoEval.stats[1]  # update mAP to pycocotools mAP
    #
    # # Return results
    # return mp, mr, map, mf1, loss / len(dataloader)
