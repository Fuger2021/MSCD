import torch, os, glob, cv2, random
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from MSCD.models.unet import Net
from MSCD.utils.utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Initialization for DDP
dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Parser
parser = ArgumentParser()
parser.add_argument("--epoch",           type=int,   default=60)
parser.add_argument("--step_number",     type=int,   default=3)
parser.add_argument("--learning_rate",   type=float, default=1e-4)
parser.add_argument("--batch_size",      type=int,   default=8)
parser.add_argument("--patch_size",      type=int,   default=256)
parser.add_argument("--cs_ratio",        type=float, default=0.1)
parser.add_argument("--block_size",      type=int,   default=32)
parser.add_argument("--model_dir",       type=str,   default="weight")
parser.add_argument("--data_dir",        type=str,   default="data")
parser.add_argument("--log_dir",         type=str,   default="log")
parser.add_argument("--tensorboard_dir", type=str,   default="tensorboard_logs")
parser.add_argument("--save_interval",   type=int,   default=5)
parser.add_argument("--trainset_name",   type=str,   default="pristine_images")
parser.add_argument("--testset_name",    type=str,   default="Set11")

args = parser.parse_args()

# Basic Configs
seed = 2025 + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
epoch = args.epoch
learning_rate = args.learning_rate
T = args.step_number
B = args.block_size
global_batch_size = args.batch_size
local_batch_size = global_batch_size // world_size
bsz = local_batch_size
psz = args.patch_size
ratio = args.cs_ratio
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

# Logger
if rank == 0:
    assert global_batch_size % world_size == 0

    tensorboard_exp_dir = os.path.join(
        args.tensorboard_dir, 
        f"R_{ratio}_T_{T}_B_{B}"
    )
    os.makedirs(tensorboard_exp_dir, exist_ok=True)

    writer = SummaryWriter(tensorboard_exp_dir)
    writer.add_text(
        'Hyperparameters', 
        f"cs_ratio: {ratio}\n"
        f"step_number: {T}\n"
        f"block_size: {B}\n"
        f"learning_rate: {learning_rate}\n"
        f"total_batch_size: {global_batch_size}\n"
        f"batch_size_per_gpu: {bsz}\n"
        f"num_gpus: {world_size}\n"
        f"patch_size: {psz}\n"
        f"testset: {args.testset_name}"
    )

    print("CS Ratio =", ratio)
    print("Batch Size(/GPU) =", bsz)
    print("Patch Size =", psz)
    print("Number of GPUs =", world_size)

# Training Configs
iter_num = 1000
N = B * B
q = int(np.ceil(ratio * N))

# Apply the same Phi across all GPUs
if rank == 0:
    U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
    Phi = (U @ V)[:, :q].contiguous()
else:
    Phi = torch.empty(N, q, device=device)
dist.broadcast(Phi, src=0)

start_time = time()
training_image_paths = glob.glob(os.path.join(args.data_dir, args.trainset_name) + "/*")

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)
model = DDP(Net(T, pipe.unet).to(device), device_ids=[rank])
model._set_static_graph()

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")

class MyDataset(Dataset):
    def __getitem__(self, index):
        while True:
            path = random.choice(training_image_paths)
            x = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
            x = torch.from_numpy(x[:, :, 0]) / 255.0
            h, w = x.shape
            max_h, max_w = h - psz, w - psz
            if max_h < 0 or max_w < 0:
                continue
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)
            return x[start_h:start_h+psz, start_w:start_w+psz]

    def __len__(self):
        # iter_num * bsz per GPU
        return iter_num * bsz * world_size

train_dataset = MyDataset()
train_sampler = DistributedSampler(
    train_dataset, 
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed
)

dataloader = DataLoader(
    train_dataset,
    batch_size=bsz,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

model_dir = "./%s/R_%.2f_T_%d_B_%d" % (args.model_dir, ratio, T, B)
log_path = "./%s/R_%.2f_T_%d_B_%d.txt" % (args.log_dir, ratio, T, B)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name, "*"))

def test():
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i, path in enumerate(test_image_paths):
            test_image = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image[:,:,0], block_size=B)
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
            x = torch.from_numpy(img_pad).to(device).float()
            perm = torch.randperm(new_h * new_w, device=device)
            perm_inv = torch.empty_like(perm)
            perm_inv[perm] = torch.arange(perm.shape[0], device=device)
            A = lambda z: (z.reshape(-1,)[perm].reshape(-1,N) @ Phi)
            AT = lambda z: (z @ Phi.t()).reshape(-1,)[perm_inv].reshape(1,1,new_h,new_w)
            y = A(x)
            x_out = model.module(y, A, AT, use_amp_=False)[..., :old_h, :old_w]
            x_out = (x_out.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()
            PSNR = psnr(x_out, img)
            SSIM = ssim(x_out, img, data_range=255)
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)

    cur_psnr, cur_ssim = np.mean(PSNR_list), np.mean(SSIM_list)
    
    if rank == 0:
        log_data = "CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f." % (ratio, cur_psnr, cur_ssim)
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        return cur_psnr, cur_ssim

    return None, None

if rank == 0:
    print("============== Start Training ==============")

for epoch_i in range(1, epoch + 1):
    train_sampler.set_epoch(epoch_i)
    start_time = time()
    loss_avg = 0.0

    # Sync all GPUs
    dist.barrier()
    
    for x in tqdm(dataloader, disable=rank!=0):
        x = x.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))
        perm = torch.randperm(psz*psz, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.shape[0], device=device)
        A = lambda z: (z.reshape(bsz,-1)[:,perm].reshape(bsz,-1,N) @ Phi)
        AT = lambda z: (z @ Phi.t()).reshape(bsz,-1)[:,perm_inv].reshape(bsz,1,psz,psz)
        y = A(x)
        x_out = model(y, A, AT)
        loss = (x_out - x).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # grad clip
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), 
            max_norm=2.0,
            norm_type=2,
            error_if_nonfinite=False
        )

        scaler.step(optimizer)
        scaler.update()
        loss_avg += loss.item()
    
    loss_avg /= len(dataloader)
    loss_tensor = torch.tensor([loss_avg], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_avg = loss_tensor.item() / world_size
    
    scheduler.step()
    
    log_data = "[%d/%d] Average loss: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, epoch, loss_avg, time() - start_time, scheduler.get_last_lr()[0])
    cur_psnr, cur_ssim = test()

    if rank == 0:
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        
        # Tensorboard
        writer.add_scalar('Training/Loss', loss_avg, epoch_i)
        writer.add_scalar('Training/LearningRate', scheduler.get_last_lr()[0], epoch_i)
        writer.add_scalar('Evaluation/PSNR', cur_psnr, epoch_i)
        writer.add_scalar('Evaluation/SSIM', cur_ssim, epoch_i)
        
        if epoch_i % args.save_interval == 0:
            torch.save(model.module.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))

# Tensorboard
if rank == 0:
    writer.close()
    print("============== Training Finished ==============")
