import os
import time
import torch
import itertools
import torchvision
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def gen_save_examples(path, gen, cond_gen, period, name_format, epoch_offset=0, device="cpu",
        stat=None, A2B=False, **kwargs):
    def save_examples(epoch, batch):
        if batch % period != 0:
            return
        gen.eval()
        imgA, imgB = next(cond_gen)
        real, cond = (imgB, imgA) if A2B else (imgA, imgB)
        cond = cond.to(device)
        examples = gen(cond)
        if not os.path.isdir(path):
            os.mkdir(path)
        final = torch.cat((examples, cond), dim=0)
        if stat is not None:
            std, mean = stat
            final = (std*final + mean)
        fullpath = os.path.join(path, name_format.format(epoch=(epoch_offset+epoch), batch=batch))
        torchvision.utils.save_image(final.detach(), fp=fullpath, **kwargs)
        gen.train()
    return save_examples

def gen_save_model(path, gen, dis, batch_size, period, name_format, epochperiod=1):
    def save_model(epoch, batch, stats):
        if epoch % epochperiod != 0:
            return
        if batch % period != 0:
            return
        if not os.path.isdir(path):
            os.mkdir(path)
        fullname = os.path.join(path, name_format.format(epoch=epoch, batch=batch))
        torch.save({
            "epoch": epoch,
            "batch_size": batch_size,
            "dis_state_dict": dis.state_dict(),
            "dis_optim_state_dict": dis.optim.state_dict(),
            "gen_state_dict": gen.state_dict(),
            "gen_optim_state_dict": gen.optim.state_dict(),
            "stats": stats
        }, fullname)
    return save_model

def gen_get_cond(dataset, batch_size, device="cpu"):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    def get_cond():
        for imgA, imgB in itertools.cycle(test_loader):
            imgA, imgB = imgA.to(device), imgB.to(device)
            yield imgA, imgB
    return get_cond()

def load_model(path, gen, dis, device="cpu"):
    state = torch.load(path, map_location=torch.device(device))
    gen.load_state_dict(state["gen_state_dict"])
    gen.optim.load_state_dict(state["gen_optim_state_dict"])
    dis.load_state_dict(state["dis_state_dict"])
    dis.optim.load_state_dict(state["dis_optim_state_dict"])
    return state["epoch"], state["batch_size"], state.get("stats", {"dis": { "loss": [], "prob": []}, "gen": {"loss": []}})

def show_stats(stats):
    fig, (p1) = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

    p1.plot(stats["gen"]["loss"], color="blue", label="generator train loss")
    p1.plot(stats["dis"]["loss"], color="green", label="discriminator train loss")
    p1.set_xlabel("epochs")
    p1.set_ylabel("loss")
    p1.legend()

    plt.yscale("log")
    plt.show()


def save_model(path, name, batch_size, gen, dis, stats, epochs, epoch_offset=0):
    fullname=os.path.join(path, name)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save({
        "epoch": epochs + epoch_offset,
        "batch_size": batch_size,
        "dis_state_dict": dis.state_dict(),
        "dis_optim_state_dict": dis.optim.state_dict(),
        "gen_state_dict": gen.state_dict(),
        "gen_optim_state_dict": gen.optim.state_dict(),
        "stats": stats
    }, fullname)


def train(gen, dis, loader, epochs, device="cpu", fw_snapshot=None, model_snapshot=None,
        A2B=False, k=1, k2=1):
    print(f"Using device: {device}")
    stats = {'dis': {
        'loss': [],
        'prob': []
    }, 'gen': {
        'loss': [] 
    }}
    gen.train()
    dis.train()

    real_label = torch.ones(loader.batch_size, 1).to(device)
    fake_label = torch.zeros(loader.batch_size, 1).to(device)
    if not dis.avg_outputs:
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        out_shape = dis(dummy_input, dummy_input).squeeze(0)
        real_label = torch.ones(loader.batch_size, *out_shape.shape).to(device)
        fake_label = torch.zeros(loader.batch_size, *out_shape.shape).to(device)

    try:
        for i in range(epochs):
            pbar = tqdm(enumerate(iter(loader)), total=len(loader))
            pbar.set_description("epoch %s/%s" % (i, epochs))
            dloss, pr = 0, 0
            gloss = 0
            for j, (imgA, imgB) in pbar:
                imgA, imgB = imgA.to(device), imgB.to(device)
                batch_prob, batch_dloss, batch_gloss = 0, 0, 0
                real, cond = (imgB, imgA) if A2B else (imgA, imgB)
                for _ in range(k):
                    l, p = dis.update(real, gen(cond).to(device), cond, real_label, fake_label)
                    batch_dloss += l
                    batch_prob += p

                for _ in range(k2):
                    batch_gloss += gen.update(dis, cond, real_label, real)
                batch_prob = float(torch.mean(batch_prob) / k)
                batch_dloss = float(torch.mean(batch_dloss) / k)
                batch_gloss = float(torch.mean(batch_gloss) / k2)
                dloss += batch_dloss
                gloss += batch_gloss
                pr += batch_prob
                pbar.write("D(G(z)): %.2f, generator loss: %.2f, discriminator loss: %.2f, time: %s" % 
                           (float(batch_prob), float(batch_gloss), float(batch_dloss),
                               time.strftime("%H-%M-%S", time.localtime())), end="\r")
                if fw_snapshot is not None:
                    fw_snapshot(i, j)
                if model_snapshot is not None:
                    model_snapshot(i, j, stats)
            stats['gen']['loss'].append(gloss / len(loader))
            stats['dis']['prob'].append(pr / len(loader))
            stats['dis']['loss'].append(dloss / len(loader))
    except KeyboardInterrupt:
        if model_snapshot is not None:
            model_snapshot("%s_KeyboardInterrupt" % epochs, 0, stats)
    gen.eval()
    dis.eval()
    return stats

