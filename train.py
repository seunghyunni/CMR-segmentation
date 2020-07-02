from model.modified_unet import *
from HeartMRIDataset_DICOM import *
import numpy as np
from sklearn.model_selection import train_test_split
from utils.losses import dice_loss, lovasz_hinge, binary_xloss
from utils.metrics import dice_coef
import os
import easydict

# GPU Setting

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
args = \
    easydict.EasyDict({"batch_size": 32,
                       "lr": 1e-4,
                       "epoch": 1000,
                       "start_epoch": 0,
                       "img_dir": './train_images/*.npy',
                       "label_dir": './train_labels/*.npy',
                       "save_dir": "./results",
                       "resumepath": './weights',
                       "filename": "/sample.pkl"})

# Data dir
images = glob(args.img_dir)
labels = glob(args.label_dir)

# Split Train, Validation set
train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.1)
print (len(train_images), len(train_labels), train_images[0], train_labels[0]) 
print (len(valid_images), len(valid_labels), valid_images[0], valid_labels[0])

# where to save training results
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# model to load
pkl_path = resumepath + filename

# call dataset
train_data = HeartMRIDataset_DICOM(train_images, train_labels)
train_batch = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
valid_data = HeartMRIDataset_DICOM(valid_images, valid_labels) 
valid_batch = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

# initiate model
model = nn.DataParallel(Modified2DUNet(1,1,16))
model.to(device) 

try:
    model.load_state_dict(torch.load(pkl_path, map_location=device))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function & optimizer
#loss_func = binary_xloss
# loss_func = lovasz_hinge
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
#initial settings
loss_func = dice_loss 
optimizer =torch.optim.Adam(model.parameters(),lr=args.lr, betas=[.9, .999])

decay_epochs = [160, 180]
decay_rate = .3
cur_iter = 0
warmup_iters = 1000

# training
best_score = 0.0 # Only Save Results when Score is best.
for i in range(args.start_epoch, args.epoch):
    print("Epoch [%3d/%3d]" % (i+1, epoch))

    # LR decay
    if epoch in decay_epochs:
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        print("Decaying lr to {}".format(new_lr))

    # -------------------------------- Train Dataset --------------------------------
    running_loss = 0
    running_dice = 0
    n_samples = 0 
    model.train()
    for step,(data) in enumerate(train_batch):
        # Learning Rate Warmup
        if cur_iter <= warmup_iters:
            lr = lr * cur_iter / float(warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        image, label, patient_id, file_id = data["image"], data["label"], data["patient_id"], data["file_id"]
        optimizer.zero_grad()                     
        x = Variable(image).cuda().float()
        y_ = Variable(label).cuda().float()
       
        #  predict 
        y = model(x)
        loss = loss_func(y,y_)
        loss.backward()
        optimizer.step()
        
        cur_iter +=1

        dice_score = dice_coef(y, y_)
        
        n_batch_samples = int(image.size()[0])
        n_samples += n_batch_samples
        running_loss += loss.data * n_batch_samples
        running_dice += dice_score.data * n_batch_samples
        
        if (step == 0) or (step+1) % 100 == 0:
            print('     > Step [%3d/%3d] Loss %.4f - Dice Coef %.4f' % (step+1, len(train_batch), running_loss/n_samples, running_dice/n_samples))
            
    train_loss = running_loss / n_samples 
    train_dice = running_dice / n_samples
    
    # ----------------------------- Validation Dataset -----------------------------
    running_loss = 0
    running_dice = 0
    n_samples = 0 
    model.eval()
    with torch.no_grad():
        for step,(data) in enumerate(valid_batch):
            image, label, patient_id, file_id = data["image"], data["label"], data["patient_id"], data["file_id"]
                                                                                                    
            optimizer.zero_grad()                     
            x = Variable(image).cuda()
            y_ = Variable(label).cuda().float()

            # predict 
            y = model(x)

            loss = loss_func(y,y_)

            dice_score = dice_coef(y, y_)
            
            n_batch_samples = int(image.size()[0])
            n_samples += n_batch_samples
            running_loss += loss.data *n_batch_samples
            running_dice += dice_score.data *n_batch_samples
            
    valid_loss = running_loss / n_samples
    valid_dice = running_dice / n_samples
    print('     ==> Result : Train Loss %.4f - Train Dice Coef %.4f - Valid Loss %.4f - Valid Dice Coef %.4f\n'
                % (train_loss, train_dice, valid_loss, valid_dice))
    
    if valid_dice > best_score:
        best_score = valid_dice
        # save results
        pkl_save_dir = os.path.join(save_dir, 'pkl')
        img_save_dir = os.path.join(save_dir, 'example_img')
        if not os.path.exists(pkl_save_dir):
            os.makedirs(pkl_save_dir)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        file_id = file_id[0]
        patient_id = patient_id[0]
         
        # save image 
        batch_tensor = torch.cat((x, y_, y), dim=3)
        grid_img = v_utils.make_grid(batch_tensor, padding=1, nrow = 4)
        v_utils.save_image(grid_img, save_dir+"/example_img/original_label_gen_image_pt_{}_file_{}_epoch_{}_dice_{:.4f}.png".format(patient_id, file_id, i+start_epoch+1, valid_dice))
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'pkl', 'epoch_%d_lr_%.5f_loss_%.4f_dice_%.4f.pkl' % ((i+start_epoch+1), lr, valid_loss, valid_dice)))
