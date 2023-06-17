import torch
import torch.nn.functional as F


def cal_deg_loss(args, opts, logits):
    deg_loss = 0.0

    for i in range(args.batch_size):
        pos = opts[i]
        deg_loss += (-torch.log(logits[i][pos]))
    
    deg_loss /= args.batch_size

    return deg_loss


def cal_trp_loss(args, original_features, degraded_features, trp1_features, trp2_features, opts):
    trp_loss = 0.0

    original_features = F.normalize(original_features, p=2, dim=[1, 2, 3])
    degraded_features = F.normalize(degraded_features, p=2, dim=[1, 2, 3])
    trp1_features = F.normalize(trp1_features, p=2, dim=[1, 2, 3])
    trp2_features = F.normalize(trp2_features, p=2, dim=[1, 2, 3])

    diff_1 = original_features-degraded_features
    diff_2 = original_features-trp1_features
    diff_3 = original_features-trp2_features

    degraded_dis = torch.sum(torch.square(diff_1), dim=[1, 2, 3])
    trp1_dis = torch.sum(torch.square(diff_2), dim=[1, 2, 3])
    trp2_dis = torch.sum(torch.square(diff_3), dim=[1, 2, 3])

    trp_num = 0

    for i in range(args.batch_size):
        if opts[i] <= 26:
            trp_num += 1
            if opts[i] % 3 == 0:
                trp_loss_1 = degraded_dis[i] - trp1_dis[i] + 1
                trp_loss_2 = trp1_dis[i] - trp2_dis[i] + 1
                loss = max(0, trp_loss_1) + max(0, trp_loss_2) - 1
                trp_loss += max(loss, 0)
            elif opts[i] % 3 == 1:
                trp_loss_1 = trp1_dis[i] - degraded_dis[i] + 1
                trp_loss_2 = degraded_dis[i] - trp2_dis[i] + 1
                loss = max(0, trp_loss_1) + max(0, trp_loss_2) - 1
                trp_loss += max(loss, 0)
            else:
                trp_loss_1 = trp1_dis[i] - trp2_dis[i] + 1
                trp_loss_2 = trp2_dis[i] - degraded_dis[i] + 1
                loss = max(0, trp_loss_1) + max(0, trp_loss_2) - 1
                trp_loss += max(loss, 0)
        
    trp_loss /= max(trp_num, 1)

    return trp_loss
