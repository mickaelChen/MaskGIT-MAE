# simple scrip to download the pretrained model
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/last.ckpt", local_dir=".")
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/VQGAN/model.yaml", local_dir=".")
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256_FID_6_80.pth", local_dir=".")
hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_512_FID_7_65.pth", local_dir=".")
