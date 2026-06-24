# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import datetime
import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo4 import YoloBody
from tqdm.auto import tqdm


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, scaler):
    total_loss = 0
    val_loss = 0
    _epoch_start = time.time()
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            with autocast():
                outputs = net(images)
                losses = []
                for i in range(len(outputs)):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                loss = sum(losses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                with autocast():
                    outputs = net(images_val)
                    losses = []
                    for i in range(len(outputs)):
                        loss_item = yolo_losses[i](outputs[i], targets_val)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    train_loss  = total_loss / (epoch_size + 1)
    v_loss      = val_loss   / (epoch_size_val + 1)
    elapsed     = time.time() - _epoch_start
    remaining   = elapsed * (Epoch - epoch - 1)
    eta_str     = str(datetime.timedelta(seconds=int(remaining)))

    print('\n' + '='*60)
    print(f'  Epoch      : {epoch+1:>4d} / {Epoch}')
    print(f'  Train Loss : {train_loss:.6f}')
    print(f'  Val   Loss : {v_loss:.6f}')
    print(f'  LR         : {get_lr(optimizer):.2e}')
    print(f'  Epoch Time : {elapsed:.1f}s   |   ETA: {eta_str}')
    print('='*60 + '\n')

    if not os.path.exists(weights_save_dir):
        os.makedirs(weights_save_dir)
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'{weights_save_dir}/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
            (epoch + 1), train_loss, v_loss))


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # ----------------------------------------------------#
    #   Auto-setup model_data and lzsp_train20201202.txt
    # ----------------------------------------------------#
    import os
    from PIL import Image

    model_data_dir = 'model_data'
    os.makedirs(model_data_dir, exist_ok=True)
    
    classes = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]
    with open(os.path.join(model_data_dir, 'single_cell.txt'), 'w') as f:
        f.write('\n'.join(classes) + '\n')
    with open(os.path.join(model_data_dir, 'lzsp_classes.txt'), 'w') as f:
        f.write('\n'.join(classes) + '\n')
        
    custom = True # Set to True for MSA-YOLO
    
    if custom:
        anchors_list = "5,6, 8,11, 10,14, 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401"
    else:
        anchors_list = "12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401"
        
    with open(os.path.join(model_data_dir, '608_anchors.txt'), 'w') as f:
        f.write(anchors_list)
    with open(os.path.join(model_data_dir, 'yolo_anchors.txt'), 'w') as f:
        f.write(anchors_list)
        
    annotation_path = 'lzsp_train20201202.txt'
    if not os.path.exists(annotation_path):
        print("Dataset annotations not found. Auto-generating lzsp_train20201202.txt...")
        if os.path.exists('sipakmed_data'):
            lines = []
            categories = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
            for idx, cat in enumerate(categories):
                cat_path = os.path.join('sipakmed_data', cat, "CROPPED")
                if not os.path.exists(cat_path):
                    cat_path = os.path.join('sipakmed_data', cat)
                if os.path.exists(cat_path):
                    files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
                    print(f"Adding category: {cat} ({len(files)} files)")
                    for file in files:
                        img_path = os.path.join(cat_path, file)
                        try:
                            with Image.open(img_path) as im:
                                w, h = im.size
                            lines.append(f"{img_path} 0,0,{w},{h},{idx}\n")
                        except:
                            pass
            with open(annotation_path, 'w') as f:
                f.writelines(lines)
            print(f"Generated lzsp_train20201202.txt with {len(lines)} annotations!")
        else:
            print("Error: sipakmed_data folder not found! Please check dataset path.")

    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (608, 608)
    # -------------------------------#
    #   tricks的使用设置
    # -------------------------------#
    Cosine_lr = True
    mosaic = True
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = 0
    # custom variable moved up
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True

    annotation_path = 'lzsp_train20201202.txt'
    # -------------------------------#
    #   获得先验框和类
    # -------------------------------#
    anchors_path = 'model_data/608_anchors.txt'
    classes_path = 'model_data/single_cell.txt'
    weights_save_dir = "trained_weights/msa_yolo" if custom else "trained_weights/baseline_yolov4"
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes, custom=custom)
    # -------------------------------------------#
    #   权值文件的下载请看README
    # -------------------------------------------#
    model_path = ""
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    
    # Load state dict (check shapes dynamically to handle custom/baseline differences)
    try:
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    except Exception as e:
        print(f"Skipping loading weights: {e}")
        
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    scaler = GradScaler()

    yolo_losses = []
    num_heads = 4 if custom else 3
    all_anchors = np.reshape(anchors, [-1, 2])
    anchors_per_head = len(all_anchors) // num_heads
    for i in range(num_heads):
        head_anchors = all_anchors[i * anchors_per_head:(i + 1) * anchors_per_head]
        yolo_losses.append(YOLOLoss(head_anchors, num_classes, \
                                    (input_shape[1], input_shape[0]), Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.3
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 8  # Optimized for Colab T4 16GB VRAM
        Init_Epoch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda, scaler)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 8  # Optimized for Colab T4 16GB VRAM
        Freeze_Epoch = 50
        Unfreeze_Epoch = 200

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda, scaler)
            lr_scheduler.step()
