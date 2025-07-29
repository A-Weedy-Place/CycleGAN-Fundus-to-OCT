import os
import argparse
import itertools
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

# ---------------------- Dataset ----------------------
class UnpairedImageDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None):
        self.filesA = sorted(os.listdir(rootA))
        self.filesB = sorted(os.listdir(rootB))
        self.rootA, self.rootB = rootA, rootB
        self.transform = transform

    def __len__(self):
        return max(len(self.filesA), len(self.filesB))

    def __getitem__(self, idx):
        imgA = Image.open(os.path.join(self.rootA, self.filesA[idx % len(self.filesA)])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootB, self.filesB[idx % len(self.filesB)])).convert('RGB')
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return {'A': imgA, 'B': imgB}

# ---------------------- Models ----------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        dim = 64
        for _ in range(2):  # downsample
            layers += [
                nn.Conv2d(dim, dim*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(True)
            ]
            dim *= 2
        for _ in range(n_blocks):  # residual blocks
            layers += [ResnetBlock(dim)]
        for _ in range(2):  # upsample
            layers += [
                nn.ConvTranspose2d(dim, dim//2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(dim//2),
                nn.ReLU(True)
            ]
            dim //= 2
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_ch, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        dim = 64
        for mult in [2, 4, 8]:
            layers += [
                nn.Conv2d(dim, dim*mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim*mult),
                nn.LeakyReLU(0.2, True)
            ]
            dim *= mult
        layers += [nn.Conv2d(dim, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------------------- Training ----------------------
def train(args):
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()

    # DataLoader setup
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = UnpairedImageDataset(args.dataA, args.dataB, tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        persistent_workers=True
    )

    # Instantiate models
    G_AB = Generator(3, 3).to(args.device)
    G_BA = Generator(3, 3).to(args.device)
    D_A  = Discriminator(3).to(args.device)
    D_B  = Discriminator(3).to(args.device)

    # Optimizers & Losses
    opt_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    criterion_GAN   = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_id    = nn.L1Loss()

    # Resume logic
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        G_AB.load_state_dict(ckpt['G_AB'])
        G_BA.load_state_dict(ckpt['G_BA'])
        D_A.load_state_dict(ckpt['D_A'])
        D_B.load_state_dict(ckpt['D_B'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Using device: {args.device}")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        for i, batch in enumerate(loader):
            realA = batch['A'].to(args.device, non_blocking=True)
            realB = batch['B'].to(args.device, non_blocking=True)

            # Generator update
            with autocast():
                idA = G_BA(realA); idB = G_AB(realB)
                loss_id = criterion_id(idA, realA) + criterion_id(idB, realB)
                fakeB = G_AB(realA); fakeA = G_BA(realB)
                loss_GAN_A2B = criterion_GAN(D_B(fakeB), torch.ones_like(D_B(fakeB)))
                loss_GAN_B2A = criterion_GAN(D_A(fakeA), torch.ones_like(D_A(fakeA)))
                recA = G_BA(fakeB); recB = G_AB(fakeA)
                loss_cycle = criterion_cycle(recA, realA) + criterion_cycle(recB, realB)
                loss_G = (loss_GAN_A2B + loss_GAN_B2A
                         + args.lambda_cycle * loss_cycle
                         + args.lambda_id * loss_id)
            opt_G.zero_grad(); scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()

            # Discriminator update
            with autocast():
                pred_realB = D_B(realB); pred_fakeB = D_B(fakeB.detach())
                loss_D_B = 0.5 * (
                    criterion_GAN(pred_realB, torch.ones_like(pred_realB))
                    + criterion_GAN(pred_fakeB, torch.zeros_like(pred_fakeB))
                )
                pred_realA = D_A(realA); pred_fakeA = D_A(fakeA.detach())
                loss_D_A = 0.5 * (
                    criterion_GAN(pred_realA, torch.ones_like(pred_realA))
                    + criterion_GAN(pred_fakeA, torch.zeros_like(pred_fakeA))
                )
                loss_D = loss_D_A + loss_D_B
            opt_D.zero_grad(); scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()

            # Logging
            if (i + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                its = args.log_interval
                print(f"Epoch[{epoch}/{args.epochs}] Step[{i+1}/{len(loader)}] "
                      f"G:{loss_G.item():.4f} D:{loss_D.item():.4f} "
                      f"({its/elapsed:.2f} iters/sec, {elapsed:.1f}s/{its} iters)")
                start_time = time.time()

        # Save sample images
        with torch.no_grad():
            sampleA = realA[:4]; genB = G_AB(sampleA)
            save_image(torch.cat((sampleA, genB), 0),
                       os.path.join(args.output, f"epoch{epoch}_sample.jpg"),
                       nrow=4, normalize=True)
        # Checkpoint every 5 epochs
        if epoch % args.save_interval == 0:
            ckpt = {
                'epoch': epoch,
                'G_AB': G_AB.state_dict(), 'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),   'D_B': D_B.state_dict(),
                'opt_G': opt_G.state_dict(),'opt_D': opt_D.state_dict()
            }
            torch.save(ckpt, os.path.join(args.output, f"checkpoint_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataA", default="data/fundus/trainA")
    parser.add_argument("--dataB", default="data/oct/trainB")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size",  type=int, default=128)
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_id",    type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--save_interval",  type=int, default=5)
    parser.add_argument("--resume",        type=str,   default="")
    parser.add_argument("--device",        type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train(args)
