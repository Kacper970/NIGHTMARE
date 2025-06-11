import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

sys.path.append(".")
from engine import eval_one_epoch
from models import MaskedSequenceEncoder
from data import SequenceCandidateDataset, CandidateSampler

class ModalityProcessor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.text_mlp2 = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, text_emb2,  vision_emb):
        t_out = self.text_mlp(text_emb)
        t_out2 = self.text_mlp2(text_emb2)
        v_out = self.vision_mlp(vision_emb)
        return t_out, t_out2, v_out
        
        
class ModalityProcessor3(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.text_mlp2 = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, text_emb2,  vision_emb):
        t_out = self.text_mlp2(self.text_mlp(text_emb))
        t_out2 = self.text_mlp(self.text_mlp2(text_emb2))
        v_out = self.vision_mlp(vision_emb)
        return t_out, t_out2, v_out
        
class ModalityProcessor7(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.text_mlp2 = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, text_emb2,  vision_emb):
    
        t_out = self.text_mlp2(self.text_mlp(text_emb))
        
        t_out2 = self.text_mlp2(self.text_mlp(text_emb2))
        
        v_out = self.vision_mlp(vision_emb)
        
        return t_out, t_out2, v_out
        
        
class ModalityProcessor4(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, text_emb2,  vision_emb):
        t_out = self.text_mlp(text_emb + text_emb2)
        v_out = self.vision_mlp(vision_emb)
        return t_out, v_out
        
class ModalityProcessor5(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.text_mlp2 = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)
        self.final = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, text_emb2,  vision_emb):
        t_out = self.text_mlp(text_emb)
        t_out2 = self.text_mlp2(text_emb2)
        v_out = self.vision_mlp(vision_emb)
        t_out_final = self.final(t_out + t_out2)
        
        return t_out_final, v_out
        
class ModalityProcessor6(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.text_mlp2 = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)
        self.final = nn.Bilinear(in1_features=384, in2_features=384, out_features=384)

    def forward(self, text_emb, text_emb2,  vision_emb):
        t_out = self.text_mlp(text_emb)
        t_out2 = self.text_mlp2(text_emb2)
        v_out = self.vision_mlp(vision_emb)
        t_out_final = self.final(t_out, t_out2)
        
        return t_out_final, v_out


class FILMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, features, condition_input):
        gamma = self.gamma_fc(condition_input)  # (B, 1, D)
        beta = self.beta_fc(condition_input)    # (B, 1, D)
        return gamma * features + beta
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))
        
        self.embeddings = nn.Parameter(torch.randn(max_seq_length, d_model))

    def forward(self, x):
        #return x + self.pe[:, : x.size(2)]
        
        return self.embeddings + x + self.pe

class GatedMLP(nn.Module):

    def __init__(self, dim=300, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        gated = torch.sigmoid(self.gate(x))
        x = self.act(self.fc1(x)) * gated
        x = self.fc2(x)
        return x
        
class ModalityProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_mlp = GatedMLP(dim=384, hidden=512)
        self.vision_mlp = GatedMLP(dim=384, hidden=512)

    def forward(self, text_emb, vision_emb):
        t_out = self.text_mlp(text_emb)
        v_out = self.vision_mlp(vision_emb)
        return t_out, v_out

class Combinerpaper(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combinerpaper, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        #target_features = F.normalize(target_features, dim=-1)

        #logits = self.logit_scale * predicted_features @ target_features.T
        
        return predicted_features

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)

class SequenceCombiner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SequenceCombiner, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # Combine both sequences
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence, sen_emb):
        # Concatenate the sequences along the last dimension (feature dimension)
        combined = torch.cat((sequence, sen_emb), dim=-1)
        #combined = combined.to(device)
        x = torch.relu(self.fc1(combined))
        return self.fc2(x)
        
class AttentionCombiner(nn.Module):
    def __init__(self, input_dim):
        super(AttentionCombiner, self).__init__()
        self.attention = nn.Parameter(torch.randn(input_dim))  # learnable attention vector

    def forward(self, sequence, sen_emb):
        # Compute attention scores (dot product) for each element
        attention_scores = torch.sum(sequence * self.attention, dim=-1) + torch.sum(sen_emb * self.attention, dim=-1)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Weighted sum of the sequences using attention scores
        combined = sequence * attention_scores.unsqueeze(-1) + sen_emb * (1 - attention_scores).unsqueeze(-1)
        return combined


class ExperimentTracker:
    def __init__(self, num_candidates_steps):
        self.num_candidates_steps = num_candidates_steps

        self.step = 0
        self.top1_accuracies = []
        self.top5_accuracies = []
        self.top10_accuracies = []

    def log(self, dict):
        self.top1_accuracies.append(dict["val_top1_accuracy"])
        self.top5_accuracies.append(dict["val_top5_accuracy"])
        self.top10_accuracies.append(dict["val_top10_accuracy"])
        self.step += 1

    def print_results(self):
        table = [
            [
                self.num_candidates_steps[i],
                f"{self.top1_accuracies[i] * 100:.2f}%",
                f"{self.top5_accuracies[i] * 100:.2f}%",
                f"{self.top10_accuracies[i] * 100:.2f}%",
            ]
            for i in range(self.step)
        ]

        print(
            tabulate(
                table,
                headers=[
                    "Num Candidates",
                    "Top-1 Accuracy",
                    "Top-5 Accuracy",
                    "Top-10 Accuracy",
                ],
                tablefmt="fancy_grid",
            )
        )


def get_dataloader(num_candidates, args):
    transforms = T.Compose(
        [
            T.Lambda(lambda x: torch.from_numpy(x).float()),
        ]
    )

    candidate_sampler = CandidateSampler(
        args.data_dir,
        split=args.split,
        source=args.source,
        modality=args.modality,
        sampling_key="panel_level",
        transform=transforms,
        num_candidates=num_candidates - 1,
    )

    dataset = SequenceCandidateDataset(
        args.data_dir,
        candidate_sampler,
        split=args.split,
        modality=args.modality,
        source=args.source,
        transform=transforms,
        pre_seq_len=args.pre_seq_len,
        suf_seq_len=args.suf_seq_len,
        return_metadata=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_split_text_dialog",)
    parser.add_argument("--source", default="features")
    parser.add_argument("--modality", default="text")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-path", default="models/checkpoint_NIGHTMARE, (3 Channels).pt")

    parser.add_argument("--feat-dim", type=int, default=1152)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--pre-seq-len", type=int, default=2)
    parser.add_argument("--suf-seq-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument(
        "--num-candidates-steps", nargs="+", type=int, default=[50, 200, 500, 1000]
    )
    args = parser.parse_args()

    args.track_failure_cases = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_model:
        
        model = MaskedSequenceEncoder(
            input_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        model_head = nn.Linear(args.hidden_dim, args.feat_dim)
        
        #COMBINER = SequenceCombiner(384, 768, 384)
        #COMBINER = AttentionCombiner(384)
        #COMBINER = Combinerpaper(384, 768, 384)
        #bart = PositionalEncoding(1152, 1)
        #film = FILMLayer(384, 384)       
        #COMBINER.load_state_dict(checkpoint["combiner"])
        #bart.load_state_dict(checkpoint["bart"])
        #film.load_state_dict(checkpoint["film"])
        

        channel_proc = ModalityProcessor2()
        

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model_head.load_state_dict(checkpoint["model_head_tok"])
        channel_proc.load_state_dict(checkpoint["channel"])
        
        
    else:
        
        if args.source == "features":
            model = nn.Identity()
            model_head = nn.Identity()

            model.forward = (
                lambda x: (
                    x[:, : args.pre_seq_len, :].mean(dim=1)
                    + x[:, -args.suf_seq_len :, :].mean(dim=1)
                )
                / 2
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_tracker = ExperimentTracker(args.num_candidates_steps)
    best_acc = {"top1": 0, "top5": 0, "top10": 0}

    for num_candidates in args.num_candidates_steps:
        args.num_eval_candidates = num_candidates
        dataloader = get_dataloader(num_candidates, args)
        eval_one_epoch(
            model,
            model_head,
            dataloader,
            device,
            num_candidates,
            experiment_tracker,
            best_acc,
            args, 5, channel_proc, 5
        )

        experiment_tracker.print_results()
