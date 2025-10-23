"""
Self Supervised Pretraining for Image Classification with Vision Transformers
modified to use for ResNet this time

Following the approach from MONAI: https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining/vit_unetr_ssl
we simply use all CTA images, thus not limited to data fulfilling the inclusion criteria of the study (RFS times)
"""

#Setup imports
import os
import json
import time
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from ResNetAutoEnc import ResNetAutoEnc
#from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

import numpy as np

print_config()

# some parameters
use_CTA_only = True
skip_multiple_CTAs = True
image_size = (224, 224, 320)  # (x, y, z) dimensions for the input images

#------------Define file paths & output directory path------------
#pretraining path: this path + "pretraining"
this_file_path = os.path.dirname(os.path.abspath(__file__))
pretrain_path = os.path.join(this_file_path, "pretraining")
if not os.path.isdir(pretrain_path):
    os.makedirs(pretrain_path)

model_path = os.path.join(pretrain_path, "model_weights")
if not os.path.isdir(model_path):
    os.makedirs(model_path)

#chose directory:
data_dir = "/home/christian/data/TirolKliniken/Anonymized_Data"    #Server
if not os.path.isdir(data_dir):
    print(f"{data_dir} does not exist. EXIT.")
    exit()

blocked_train_items_path = os.path.join(this_file_path, "regression/run_25/val_data_selected.txt")

def create_dicom_folder_list(dir):
    """
    returns a list of all folders in the given directory
    assuming directory to be "1st_stroke_event", "2nd_stroke_event" or "3rd_stroke_event" ...
    """
    if not os.path.isdir(dir):
        print(f"{dir} is empty. returning empty dir [] .")
        return []   #empty directory
    imageset_dir0 = [os.path.join(dir,d) for d in next(os.walk(dir))[1]] #M/F
    data_folders = []
    for dir1 in imageset_dir0:   #patientID
        imageset_dir1 = [os.path.join(dir1,d) for d in next(os.walk(dir1))[1]]
        for dir2 in imageset_dir1: #studyID
            imageset_dir2 = [os.path.join(dir2,d) for d in next(os.walk(dir2))[1]]
            for dir3 in imageset_dir2: #seriesNr
                imageset_dir3 = [os.path.join(dir3,d) for d in next(os.walk(dir3))[1]]
                #print("imageset_dir3",imageset_dir3)
                data_folders += (imageset_dir3)
    #print(data_folders) #contains last folder of each patient that contains the dicom files
    return data_folders

img_dirs_1st = create_dicom_folder_list(f"{data_dir}/1st_stroke_event")
img_dirs_more = create_dicom_folder_list(f"{data_dir}/2nd_stroke_event")
img_dirs_more += create_dicom_folder_list(f"{data_dir}/3rd_stroke_event")
#...
print("len(img_dirs_1st): ", len(img_dirs_1st))
print("first five items in images_dir: ", img_dirs_1st[:5])

#--------------------PREPROCESSING DATA--------------------#
def get_CTA_images(img_dirs):
    # in series_desciption: "CTA" is the keyword for the images
    # we have already the file "series_description.txt" with the following format:
    #seriesNr0: CTA
    #seriesNr1: nativ with seriesNr* being the folder name of the folder containing the dicoms
    img_dirs_CTA = []
    seriesNr_dir = "" # for first check
    patID_dir = "" # for first check
    patients_without_CTA = []
    for imdir in img_dirs:
        if skip_multiple_CTAs:
            #then check if already added a CTA of this patient
            if "/".join(imdir.split("/")[:-2]) == patID_dir: #same as old patID_dir, can happen, when more seriesNr_dirs available per patient
                #print(f"Already added a CTA of this patient {patID_dir}. skipped.")
                continue #skip as already added a CTA of this patient
        if "/".join(imdir.split("/")[:-1]) == seriesNr_dir: #same as old seriesNr_dir
            #print(f"Already processed the folder {seriesNr_dir}. skipped.")
            continue #skip as already processed the folder
        seriesNr_dir = "/".join(imdir.split("/")[:-1])
        with open(seriesNr_dir + "/series_description.txt", 'r') as f:
            series_description = f.read()
            #find the line with "CTA" in it and return the seriesNr
            seriesNr = [line.split(":")[0] for line in series_description.split("\n") if "CTA" in line]
            if len(seriesNr) <1:
                #print(f"no CTA found in {seriesNr_dir}. skipped.")
                patients_without_CTA.append(seriesNr_dir)
                #dont update patID_dir, as we want to add the first CTA of the patient
                continue
            elif len(seriesNr) > 1:     #TODO: Beware what to do here.. use only one of them? use all? --> but then have different probs for one patient !!
                if skip_multiple_CTAs: #add only first CTA
                    #print(f"multiple CTA found in {seriesNr_dir}: {seriesNr[0]}. added.")
                    img_dirs_CTA.append(f"{seriesNr_dir}/{seriesNr[0]}")
                    patID_dir = "/".join(imdir.split("/")[:-2])
                else:
                    for s in seriesNr:
                        #print(f"multiple CTA found in {seriesNr_dir}: {s}")
                        img_dirs_CTA.append(f"{seriesNr_dir}/{s}")
                        patID_dir = "/".join(imdir.split("/")[:-2])
            else: #len(seriesNr) == 1
                #print("Found CTA in ", seriesNr_dir)
                seriesNr = seriesNr[0]
                img_dirs_CTA.append(f"{seriesNr_dir}/{seriesNr}")
                patID_dir = "/".join(imdir.split("/")[:-2])
                
    with open(f"{pretrain_path}/no_CTA_found.txt", 'a') as f:
        for item in patients_without_CTA:
            f.write(f"{item}\n")
    #update img_dirs_1st to img_dirs_1st_CTA
    img_dirs = img_dirs_CTA
    return img_dirs

img_dirs_1st = get_CTA_images(img_dirs_1st)
print("len(img_dirs_1st after CTA filtering): ", len(img_dirs_1st))
img_dirs_more = [] #get_CTA_images(img_dirs_more)
print("len(img_dirs_more after CTA filtering): ", len(img_dirs_more))

#define train and val data
all_data = [] #list of dictionaries with keys "image"
for imdir in img_dirs_1st + img_dirs_more:
    if not os.path.isdir(imdir):
        print(f"{imdir} is not a directory. Skipping.")
        continue
    if not os.path.isfile(os.path.join(imdir, "registeredImage.nii.gz")):
        print(f"{imdir} does not contain a registeredImage.nii.gz file. Skipping.")
        continue
    all_data.append({"image": os.path.join(imdir, "registeredImage.nii.gz")})

# Read all image paths as-is (with full filenames like registeredImage.nii.gz)
blocked_train_items = []
with open(blocked_train_items_path, "r") as f:
    for line in f:
        if line.strip():
            item = eval(line.strip())  # because it's in Python dict format, not JSON
            blocked_train_items.append(item["image"])
print("-----------------------------------------------------------------")
#print(blocked_train_items[0])
#print(all_data[0]['image'])

# Filter all_data into val and train sets
val_data = [item for item in all_data if item['image'] in blocked_train_items] #used for test data
train_data = [item for item in all_data if item['image'] not in blocked_train_items]

# Desired split: ~80% train, 20% val
split_ratio_current = len(train_data) / max(len(val_data), 1)  # Avoid divide by zero
#print(f"current split ratio is {split_ratio_current}.")

if split_ratio_current > 4:  # If train set is too large
    # Calculate how many items need to move from train to val to reach approx 80:20
    desired_val_len = (len(train_data) + len(val_data)) // 5
    n_to_move = desired_val_len - len(val_data)

    # Ensure n_to_move is positive and not more than available train_data
    n_to_move = max(0, min(n_to_move, len(train_data)))

    # Move items
    val_data += train_data[:n_to_move]
    train_data = train_data[n_to_move:]

#write out all data to json files, as list of dictionaries with keys "image"
#{'image': '/home/christian/data/TirolKliniken/Anonymized_Data/1st_stroke_event/F/268063/20190129/9'}
with open(os.path.join(pretrain_path, "train_data_pretraining.json"), "w") as f:
    for item in train_data:
        json.dump(item, f)
        f.write("\n")

with open(os.path.join(pretrain_path, "val_data_pretraining.json"), "w") as f:
    for item in val_data:
        json.dump(item, f)
        f.write("\n")

#reduce to 3 items each for testing     #TODO: delete when done with debugging
#train_data = train_data[:3]
#val_data = val_data[:3]

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")

#------------Define MONAI Transforms------------
# Define Training Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=3071,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=image_size),
        RandSpatialCropSamplesd(keys=["image"], roi_size=image_size, random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)

check_ds = Dataset(data=train_data, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image = check_data["image"][0][0]
print(f"image shape: {image.shape}")

#------------Training Configuration------------
# Training Config

# Define Network ViT backbone & Loss & Optimizer
device = torch.device("cuda:0")
model = ResNetAutoEnc(
        in_channels = 1,
        out_channels = 1,
        conv1_t_size = 7, #Attention: set to 7, not 8. size mismatch after conv1
        conv1_t_stride = 2,
        spatial_dims = 3,
)

model = model.to(device)

# Define Hyper-parameters for training loop
max_epochs = 500
val_interval = 2
batch_size = 1
lr = 1e-4
epoch_loss_values = []
step_loss_values = []
#epoch_cl_loss_values = []
epoch_recon_loss_values = []
val_loss_values = []
best_val_loss = 1000.0

recon_loss = L1Loss()
#contrastive_loss = ContrastiveLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Define DataLoader using MONAI, CacheDataset needs to be used
train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_data, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)


##------------Training loop with validation------------

training_start_time = time.time()

for epoch in range(max_epochs):
    epoch_start_time = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    #epoch_cl_loss = 0
    epoch_recon_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        start_time = time.time()

        #when contrastive loss
        #inputs, inputs_2, gt_input = (
        #    batch_data["image"].to(device),
        #    batch_data["image_2"].to(device),
        #    batch_data["gt_image"].to(device),
        #)
        inputs, gt_input = (
            batch_data["image"].to(device),
            batch_data["gt_image"].to(device),
        )
        optimizer.zero_grad()
        outputs_v1, hidden_v1 = model(inputs)
        #outputs_v2, hidden_v2 = model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        #flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss = recon_loss(outputs_v1, gt_input)
        #cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss # + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        step_loss_values.append(total_loss.item())

        # CL & Recon Loss Storage of Value
        #epoch_cl_loss += cl_loss.item()
        epoch_recon_loss += r_loss.item()

        end_time = time.time()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {total_loss.item():.4f}, "
            f"time taken: {end_time-start_time}s"
        )

    epoch_loss /= step
    #epoch_cl_loss /= step
    epoch_recon_loss /= step

    epoch_loss_values.append(epoch_loss)
    #epoch_cl_loss_values.append(epoch_cl_loss)
    epoch_recon_loss_values.append(epoch_recon_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % 100 == 0 or epoch == max_epochs - 1:
        checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(model_path, f"SSL_model_{epoch}.pt"))

    if epoch % val_interval == 0 or epoch == max_epochs - 1:
        print("Entering Validation for epoch: {}".format(epoch + 1))
        total_val_loss = 0
        val_step = 0
        model.eval()
        with torch.no_grad(): 
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                #when contrastive loss
                #inputs, inputs_2, gt_input = (
                #    val_batch["image"].to(device),
                #    val_batch["image_2"].to(device),
                #    val_batch["gt_image"].to(device),
                #)
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["gt_image"].to(device),
                )
                #print("Input shape: {}".format(inputs.shape)) #torch.Size([2, 1, 224, 224, 320])
                outputs_v1, _ = model(inputs)
                #outputs_v2, _ = model(inputs_2)
        
                flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
                #flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

                recon_val_loss = recon_loss(outputs_v1, gt_input)
                #cl_val_loss = contrastive_loss(flat_out_v1, flat_out_v2)
                val_loss = recon_val_loss # + cl_val_loss * recon_val_loss

                total_val_loss += val_loss.item()
                end_time = time.time()

        total_val_loss /= val_step
        val_loss_values.append(total_val_loss)
        print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

        if total_val_loss < best_val_loss:
            print(f"Saving new model based on validation loss {total_val_loss:.4f}")
            best_val_loss = total_val_loss
            checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(model_path, "SSL_model_best_val_loss.pt"))

        plt.figure(1, figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_loss_values)
        plt.grid()
        plt.title("Training Loss")

        plt.subplot(1, 2, 2)
        plt.plot(val_loss_values)
        plt.grid()
        plt.title("Validation Loss")

        #plt.subplot(2, 2, 3)
        #plt.plot(epoch_cl_loss_values)
        #plt.grid()
        #plt.title("Training Contrastive Loss")

        #plt.subplot(2, 2, 4)
        #plt.plot(epoch_recon_loss_values)
        #plt.grid()
        #plt.title("Training Recon Loss")

        plt.savefig(os.path.join(model_path, "loss_plots.png"))
        plt.savefig(os.path.join(model_path, "loss_plots.pdf"))
        plt.close(1)

        #store the plot values in files, and overwrite each val epoch with updated values
        #so the values can be read in a seperate file for plotting
        np.savetxt(os.path.join(model_path, "train_loss_values.txt"), epoch_loss_values)
        #np.savetxt(os.path.join(model_path, "train_recon_loss_values.txt"), epoch_recon_loss_values)
        np.savetxt(os.path.join(model_path, "val_loss_values.txt"), val_loss_values)

    epoch_end_time = time.time()
    total_epoch_duration = (epoch_end_time - epoch_start_time) / 60
    total_training_duration = (epoch_end_time - training_start_time) / 60

    #write to file the time:
    with open(os.path.join(model_path,"training_time_details.txt"), "a") as f:
        f.write(f"Epoch {epoch}: {total_epoch_duration:.2f} min | Cumulative: {total_training_duration:.2f} min\n")


print("Done")