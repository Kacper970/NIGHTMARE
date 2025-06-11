import torch
import numpy as np
import torch.nn.functional as F

import torch.nn as nn

from tqdm import tqdm

fmi = lambda x: x.float().mean().item()


# Define a linear layer to reduce the embedding size to 384
class DimensionalityReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
reducer = DimensionalityReducer(768, 384)
    


def train_one_epoch(
    model, model_head, dataloader, device, epoch, optimizer, logger, args, combiner, channel, bart
):
    print(f"Training epoch {epoch}")

    
    model.to(device)
    model.train()

    model_head["cls"].to(device)
    model_head["cls"].train()

    model_head["tok"].to(device)
    model_head["tok"].train()

    running_loss, running_cls_accuracy, running_tok_accuracy = 0.0, 0.0, 0.0
    step = 0
    
    if args.modality == 'text':
        
        for sequence1, sen1, sequence2, sen2, sequence3, sen3, candidates, y, sents in tqdm(dataloader):
            b, s, f = sequence1.shape
            optimizer.zero_grad()
            
            s1 = reducer(torch.tensor(np.array(sen1), dtype=torch.float32))
            s1 = s1.permute(1, 0, 2)
            s1 = s1.to(device)
            sequence1 = sequence1.to(device)
            #s1, sequence1 = channel(s1, sequence1)
            sequence1 = sequence1 + s1
            #sequence1 = bart(sequence1)
            
            s3 = reducer(torch.tensor(np.array(sen3), dtype=torch.float32))
            s3 = s3.permute(1, 0, 2)
            s3 = s3.to(device)
            sequence3 = sequence3.to(device)
            #s3, sequence3 = channel(s3, sequence3)
            sequence3 = sequence3 + s3
            #sequence3 = bart(sequence3)
            sequence = sequence2.to(device)
    
            secondary_sequence = sequence1.to(device)
            
            #x = reducer(torch.tensor(np.array(sen1), dtype=torch.float32))
            #x = x.permute(1, 0, 2)
            #secondary_sequence = x.to(device)
            
            switch_sequence = (torch.rand(b) < 0.5).to(device)
            secondary_sequence[switch_sequence] = sequence3.to(device)[switch_sequence]
            
            #xx = reducer(torch.tensor(np.array(sen3), dtype=torch.float32))
            #xx = xx.permute(1, 0, 2)
            #secondary_sequence[switch_sequence] = xx.to(device)[switch_sequence]
    
            candidates = candidates.to(device)
            y = y.to(device)
    
            mask = torch.zeros((b, s), dtype=torch.bool, device=device)
            mask[:, args.pre_seq_len] = True
            
            can_sens = reducer(torch.tensor(np.array(sents), dtype=torch.float32))
            can_sens = can_sens.permute(1, 0, 2)
            can_sens = can_sens.to(device)
            #can_sens, candidates = channel(can_sens, candidates)
            candidates = candidates + can_sens
            #candidates = bart(candidates)
            
            sen_emb = reducer(torch.tensor(np.array(sen2), dtype=torch.float32))
            sen_emb = sen_emb.permute(1, 0, 2)
            sen_emb = sen_emb.to(device)
            #sen_emb, sequence = channel(sen_emb, sequence)
            sequence = sequence + sen_emb
            #sequence = bart(sequence)
            

            #print(sequence.shape)
            #print(secondary_sequence.shape)
            x_enc = model(sequence, secondary_sequence=secondary_sequence, mask=mask)
     
            # CLS token representation to binary classification
            y_hat_cls = model_head["cls"](x_enc[:, 0]).squeeze(1).sigmoid()
    
            # Masked-token specific representation to classification
# =============================================================================
#             num_candidates = candidates.shape[1]
# =============================================================================
            tok_enc = model_head["tok"](x_enc[:, args.pre_seq_len + 1])
            # y_hat_tok = F.cosine_similarity(tok_enc.unsqueeze(1), candidates, dim=-1)
            y_hat_tok = -F.pairwise_distance(tok_enc.unsqueeze(1), candidates, p=2)
            y_hat_tok[candidates.sum(dim=-1) == 0] = -1
    
            loss = 0.0
            if not args.no_cls_optimization:
                loss += F.binary_cross_entropy(y_hat_cls, switch_sequence.float())
            if not args.no_tok_optimization:
                loss += F.cross_entropy(y_hat_tok, y)
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            running_cls_accuracy += fmi((y_hat_cls > 0.5) == switch_sequence)
            running_tok_accuracy += fmi(y_hat_tok.argmax(dim=-1) == y)
            step += 1
    
            if step % 25 == 24:
                logger.log(
                    {
                        "train_loss": running_loss / 24,
                        "train_cls_accuracy": running_cls_accuracy / 24,
                        "train_tok_accuracy": running_cls_accuracy / 24,
                        "train_epoch": epoch,
                    }
                )
    
                running_loss, running_cls_accuracy, running_tok_accuracy = 0.0, 0.0, 0.0
        
    


def eval_one_epoch(
    model, model_head, dataloader, device, epoch, logger, best_acc, args, combiner, channel, bart
):
    print(f"Evaluation epoch {epoch}")

    best_acc = best_acc.copy()
    
    
    model.to(device)
    model.eval()

    model_head.to(device)
    model_head.eval()

    loss, top1_accuracy, top5_accuracy, top10_accuracy = 0.0, 0.0, 0.0, 0.0
    failure_cases = []

    total_similarity, similarity_samples = 0.0, 0.0
    if args.modality == 'text':
        
        for _, _, sequence, sent, _, _, candidates, y, (s_comic, s_panel), (c_comics, c_panels), sents in tqdm(
            dataloader
        ):
            b, s, f = sequence.shape
            
            sequence = sequence.to(device)
            candidates = candidates.to(device)
            
            
            y = y.to(device)
    
            c_comics = np.array(c_comics)
            c_panels = np.array(c_panels)
    
            with torch.no_grad():
            
                # x_enc = model(sequence)[:, args.pre_seq_len + 1]
                
                # toevoegen zinnen
                sen_emb = reducer(torch.tensor(np.array(sent), dtype=torch.float32))
                #sen_emb = torch.tensor(np.array(sent), dtype=torch.float32)
                sen_emb = sen_emb.permute(1, 0, 2)
                sen_emb = sen_emb.to(device)
                #sen_emb, sequence = channel(sen_emb, sequence)
                sequence = sequence + sen_emb
                #sequence = bart(sequence)
                
                #x_enc = sequence[:, 1, :]
                
                can_sens = reducer(torch.tensor(np.array(sents), dtype=torch.float32))
                #can_sens = torch.tensor(np.array(sents), dtype=torch.float32)
                can_sens = can_sens.permute(1, 0, 2)
                can_sens = can_sens.to(device)
                #can_sens, candidates = channel(can_sens, candidates)
                candidates = candidates + can_sens
                #candidates = bart(candidates)
               
                
                x_enc = model(sequence)
                x_enc = model_head(x_enc[:, args.pre_seq_len + 1])
            
                
# =============================================================================
#                 num_candidates = candidates.shape[1]
# =============================================================================
    
                # y_hat = F.cosine_similarity(x_enc.unsqueeze(1), candidates, dim=-1)
                y_hat = -F.pairwise_distance(x_enc.unsqueeze(1), candidates, p=2)
                y_hat[candidates.sum(dim=-1) == 0] = -1
    
            loss += F.cross_entropy(y_hat, y).item()
            
            top1_accuracy += fmi(
                (y_hat.topk(1, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
            )
    
            top5_accuracy += fmi(
                (y_hat.topk(5, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
            )
    
            top10_accuracy += fmi(
                (y_hat.topk(10, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1)
            )
    
            for i in range(b):
                if y_hat[i].argmax() == y[i]:
                    total_similarity += y_hat[i].max().item()
                    similarity_samples += 1
    
            # Store samples where the model failed in terms of top-1 accuracy
            if args.track_failure_cases:
                for i in range(b):
                #######################################################
                    if y_hat[i].argmax() == y[i]:
                        failure_cases.append(
                            #{
                             #   "x_enc": x_enc[i],
                             #   "sequence": sequence[i],
                             #   "candidates": candidates[i],
                             #   "s_comic": s_comic[i],
                             #   "s_panel": s_panel[i],
                             #   "c_comics": c_comics[:, i],
                             #   "c_panels": c_panels[:, i],
                             #   "y": y[i],
                             #   "y_hat": y_hat[i],
                            #}
                            {
                                "s_comic": s_comic[i],
                                "s_panel": s_panel[i],
                            }

                        )
        

    best_acc["top1"] = max(best_acc["top1"], top1_accuracy / len(dataloader))
    best_acc["top5"] = max(best_acc["top5"], top5_accuracy / len(dataloader))
    best_acc["top10"] = max(best_acc["top10"], top10_accuracy / len(dataloader))

    logger.log(
        {
            "val_loss": loss / len(dataloader),
            "val_top1_accuracy": top1_accuracy / len(dataloader),
            "val_top5_accuracy": top5_accuracy / len(dataloader),
            "val_top10_accuracy": top10_accuracy / len(dataloader),
            "val_max_top1_accuracy": best_acc["top1"],
            "val_max_top5_accuracy": best_acc["top5"],
            "val_max_top10_accuracy": best_acc["top10"],
            "val_epoch": epoch,
        }
    )

    if args.track_failure_cases:
        #print(f"Average distance: {total_similarity / similarity_samples}")
        if similarity_samples > 0:
            print(f"Average distance: {total_similarity / similarity_samples}")
        else:
            print("No similarity samples to evaluate.")
        if epoch == 1000:
            torch.save(failure_cases, f"{args.checkpoint_path}_io_failure_cases_{epoch}.pt")

    return best_acc
