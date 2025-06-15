import sys
import h5py
import json
import argparse
import numpy as np
from transformers import BertTokenizer, BertModel
import torch


from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
sys.path.append(".")
#from models import MaskedSequenceEncoder
from data import SequenceClassifcationDataset
import vision_transformer as vits
import clip


# The tokenizer and embedder for BERT.
#tokenizer_text = BertTokenizer.from_pretrained('bert-base-uncased')
#model_text = BertModel.from_pretrained('bert-base-uncased')

# Using for clip (comment out the above lines when using clip instead of bert/dino):
device1 = "cuda" if torch.cuda.is_available() else "cpu"   
model_test, preprocess = clip.load("ViT-B/32", device=device1)

    
def encode(sent):

    sentence = [[x.decode('utf-8') for x in byte_str] for byte_str in sent]
    sentence = sentence[0]

    inputs = tokenizer_text(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        
        outputs = model_text(**inputs)

    last_hidden_state = outputs.last_hidden_state

    sentence_embedding = last_hidden_state.mean(dim=1).squeeze()
    
    
    # Create a mask for empty strings in 'a'
    mask = sentence == ''

    sentence_embedding[mask] = 0


    return sentence_embedding




def get_dataloaders(args):
    transforms = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    

    train_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="train",
        modality=args.modality,
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    val_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="val",
        modality=args.modality,
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    test_dataset = SequenceClassifcationDataset(
        args.data_dir,
        transform=transforms,
        split="test",
        modality=args.modality,
        source=args.source,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )






@torch.no_grad()
def extract_features(model, dataloader, device, data_path, split, modality_):
    o_file = h5py.File(f"{data_path}/{split}_features_{modality_}.hdf5", "w")

    result = {}
    res = {}
    for samples, _, comic_ids, panel_ids, sens in tqdm(dataloader):

        samples = samples.squeeze(1)
   
        samples = samples.to(device)
        
        
        # Use the following lines when working with CLIP
        features = model_test.encode_image(samples).clone().cpu().numpy()
        sens =  [[x.decode('utf-8') for x in byte_str] for byte_str in sens]
        sens = clip.tokenize(sens[0], truncate=True).to(device)
        sens_encode = model_test.encode_text(torch.tensor(sens)).clone().cpu().numpy()
        
        # Use the following lines when working with bert dino
        #features = model(samples).clone().cpu().numpy()
        #sens_encode = encode(sens).clone().cpu().numpy()

        for i, (comic_id, panel_id) in enumerate(zip(comic_ids, panel_ids)):
            
          
            
            if comic_id not in result:
                result[comic_id] = {}
                res[comic_id] = {}
            result[comic_id][panel_id] = features[i]
            res[comic_id][panel_id] = sens_encode[i]
    
    indexing = {f"{split}_features_{modality_}": {}}
    for comic_id in result:
        try:
            comic_group = o_file.create_group(f"{split}_features_{modality_}/{comic_id}")
            comic_features = np.array(list(result[comic_id].values()))
            comic_group.create_dataset("feat_data", data=comic_features)
    
            sen_features = np.array(list(res[comic_id].values()))
            comic_group.create_dataset("sen_data", data=sen_features)
    
            indexing[f"{split}_features_{modality_}"][comic_id] = {}
            indexing[f"{split}_features_{modality_}"][comic_id]["feat_data"] = list(
                result[comic_id].keys()
            )
            indexing[f"{split}_features_{modality_}"][comic_id]["sen_data"] = list(
                res[comic_id].keys()
            )
        except:
            print(comic_id)

    with open(f"{data_path}/{split}_features_{modality_}_indexing.json", "w+") as f:
        json.dump(indexing, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='Data/comics/raw_panel_images/raw_panel_images/')
    parser.add_argument("--source", default="images")
    parser.add_argument("--modality", default="text")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")

    parser.add_argument("--aug-gamma", type=float, default=0.01)
    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--pre-seq-len", type=int, default=0)
    parser.add_argument("--suf-seq-len", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
#     if not args.no_model:
#         model = MaskedSequenceEncoder(
#             input_dim=args.feat_dim,
#             hidden_dim=args.hidden_dim,
#             num_layers=args.num_layers,
#             num_heads=args.num_heads,
#             dropout=args.dropout,
#         )
# 
#         task_dim = args.hidden_dim
# 
#         checkpoint = torch.load(args.checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint["model"])
#     else:
#         if args.source == "features":
#             model = nn.Identity()
#             # model.forward = (
#             #     lambda x: (
#             #         x[:, : args.pre_seq_len, :].mean(dim=1)
#             #         + x[:, -args.suf_seq_len :, :].mean(dim=1)
#             #     )
#             #     / 2
#             # )
# 
#             model.forward = lambda x: x[:, args.pre_seq_len, :]
# 
#             task_dim = args.feat_dim
# 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
# =============================================================================
    
    
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = get_dataloaders(args)

    extract_features(model, train_dataloader, device, args.data_dir, "train", args.modality)
    extract_features(model, val_dataloader, device, args.data_dir, "val", args.modality)
    extract_features(model, test_dataloader, device, args.data_dir, "test", args.modality)
