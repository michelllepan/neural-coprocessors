import torch


def collate_data(collected):
    # Map class to tensor
    out = {}
    for label, recs in collected.items():

        t_out = torch.cat(recs, 0)
        out[label] = t_out

    return out


def rotate_dset(raw):
    # Map class to records list
    collected = {}
    for rec in raw:
        _, _, _, dout, label = rec

        label = label.item()
        if label not in collected:
            collected[label] = []
        collected[label].append(dout.unsqueeze(0))

    return collate_data(collected)


def batch_variance(batch):
    """
    For a given batch of data, we find the average of stdevs across
    feature dims and time. We use this to characterize the typical
    variance within and between object classes.
    """
    assert len(batch.shape) == 3
    std = torch.std(batch, 0)
    mean = torch.mean(std)
    mean = torch.mean(mean)
    return mean


def calc_raw_dset_class_vars(dset):
    """
    dset: a MichaelsDataset, loaded with with_labels=True
    """

    dset = rotate_dset(dset)
    stds = []
    for _, batch in dset.items():
        std = batch_variance(batch)
        stds.append(std.item())

    var_within = sum(stds) / len(stds)

    whole = torch.cat(list(dset.values()), 0)
    var_whole = batch_variance(whole).item()

    return var_whole, var_within


def calc_class_vars(actuals, labels):
    collected = {}
    for idx in range(labels.shape[0]):
        label = labels[idx].item()

        if label not in collected:
            collected[label] = []

        collected[label].append(actuals[idx : idx + 1, :, :])

    collated = collate_data(collected)

    stdevs_within = []
    for batch in collated.values():
        std = batch_variance(batch).item()
        stdevs_within.append(std)
    var_within = sum(stdevs_within) / len(stdevs_within)

    var_whole = batch_variance(actuals).item()
    return var_whole, var_within
