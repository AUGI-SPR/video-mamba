import torch
import random
import os
import argparse
from model import *
from utils.get_phases import get_phases
from batch_gen_gpt import BatchGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19990328
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


data_path = "/data2/local_datasets/"
model_path = "models"
result_path = "results"


parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train")
parser.add_argument("--dataset", default="cholec80")
parser.add_argument("--feature_extractor", default="lovit")
parser.add_argument("--mamba", action="store_true")
parser.add_argument("--causal", action="store_true")
parser.add_argument("--drop_path_rate", type=float, default=0.1)  #
parser.add_argument("--channel_mask_rate", type=float, default=0.3)  #
parser.add_argument("--lr", type=float, default=0.0005)  #
parser.add_argument("--num_epochs", type=int, default=150)
parser.add_argument("--num_layers", type=int, default=8)  #
parser.add_argument("--load_epoch", type=int, default=0)
parser.add_argument("--encoder_only", action="store_true")
parser.add_argument("--addstr", type=str, default="")
parser.add_argument("--num_f_maps", type=int, default=256)  #
parser.add_argument("--features_dim", type=int, default=768)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--sample_rate", type=int, default=1)  # 25
parser.add_argument("--r1", type=int, default=2)  #
parser.add_argument("--r2", type=int, default=2)  #
parser.add_argument("--patience", type=int, default=10)  #

args = parser.parse_args()

# args.action = "train"
# args.dataset = "cholec80"
# args.feature_extractor = "resnet"
# args.causal = False
# args.mamba = True

# args.addstr = "dp%.2f_l%d_m%.2f_lr%.4f_fm%d_r1%d_r2%d_p_%d" % (
args.addstr = "dp%.2f_l%d_m%.2f_lr%.4f_fm%d_r1%d_r2%d_p_%d" % (
    args.drop_path_rate,
    args.num_layers,
    args.channel_mask_rate,
    args.lr,
    args.num_f_maps,
    args.r1,
    args.r2,
    args.patience,
)


dataset = args.dataset
feature_extractor = args.feature_extractor
causal = args.causal
drop_path_rate = args.drop_path_rate
channel_mask_rate = args.channel_mask_rate
lr = args.lr
num_epochs = args.num_epochs
num_layers = args.num_layers
encoder_only = args.encoder_only
batch_size = args.batch_size
num_f_maps = args.num_f_maps
features_dim = args.features_dim
load_epoch = args.load_epoch
sample_rate = args.sample_rate
r1 = args.r1
r2 = args.r2
patience = args.patience
mamba = args.mamba


vid_list_file = data_path + args.dataset + "/videos/train"
vid_list_file_tst = data_path + args.dataset + "/videos/test"
features_path = data_path + args.dataset + "/features/" + feature_extractor + "/"
gt_path = data_path + args.dataset + "/groundtruth/"
mapping_file = os.path.join(data_path, args.dataset, "mapping.txt")
model_dir = (
    "./{}/".format(model_path)
    + ("ResNet-50/" if feature_extractor == "resnet" else "LoViT/")
    + ("ASMamba/" if mamba else "ASFormer/")
    + ("causal/" if causal else "bidirectional/")
    + args.dataset
    + "/"
    + args.feature_extractor
    + "/"
    + args.addstr
)
result_dir = (
    "./{}/".format(result_path)
    + ("ResNet-50/" if feature_extractor == "resnet" else "LoViT/")
    + ("ASMamba/" if mamba else "ASFormer/")
    + ("causal/" if causal else "bidirectional/")
    + args.dataset
    + "/"
    + args.feature_extractor
    + "/"
    + args.addstr
)


phases_dict, num_classes = get_phases(mapping_file)

trainer = Trainer(
    num_layers,
    r1,
    r2,
    num_f_maps,
    features_dim,
    num_classes,
    channel_mask_rate,
    mamba,
    drop_path_rate,
    args,
)

if args.action == "train":

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    batch_gen = BatchGenerator(
        num_classes, phases_dict, gt_path, features_path, sample_rate
    )
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(
        num_classes, phases_dict, gt_path, features_path, sample_rate
    )
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(
        model_dir, batch_gen, num_epochs, batch_size, lr, batch_gen_tst, patience
    )
    print("Finished training")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.predict(
        model_dir,
        result_dir,
        batch_gen_tst,
        load_epoch,
    )
    print("Finished prediction")


if args.action == "predict":

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    batch_gen_tst = BatchGenerator(
        num_classes, phases_dict, gt_path, features_path, sample_rate
    )
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(
        model_dir,
        result_dir,
        batch_gen_tst,
        load_epoch,
    )

print(args)
