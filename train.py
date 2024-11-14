import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from modules.utils import createDir
from lfista.dataset import PatchDataset
import os
from modules.DCTUtility import DCTUtility
import torch_dct as dctutil

def plot(epochs, training_loss, testing_loss, savePath=None, gui=True):
    plt.plot(training_loss, label="Training")
    plt.plot(testing_loss, label="Testing")
    plt.legend()
    plt.draw()

    if savePath!=None:
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.savefig(savePath + "losses.png")

    if not gui:
        plt.pause(1)
    plt.clf()

def phiW(x, src, hyp):
    mul_up = (dctutil.idct_2d(x.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    result = torch.mul(src, mul_up)
    return result

def phiW_delta(x, src, hyp):
    theta_m = x[:, :hyp["obsDim"], :]
    delta = x[:, hyp["obsDim"]:, :]
    mul_up = (dctutil.idct_2d(theta_m.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    result = torch.mul(src, mul_up) + delta
    return result

def phiW_additive(x, src, hyp):
    theta_m = x[:, :hyp["obsDim"], :]
    theta_a = x[:, hyp["obsDim"]:, :]
    mul_up = (dctutil.idct_2d(theta_m.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    add_up = (dctutil.idct_2d(theta_a.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    result = torch.mul(src, mul_up) + add_up
    return result

def phiW_full(x, src, hyp):
    theta_m = x[:, :hyp["obsDim"], :]
    theta_a = x[:, hyp["obsDim"]:2*hyp["obsDim"], :]
    delta = x[:, 2*hyp["obsDim"]:, :]
    mul_up = (dctutil.idct_2d(theta_m.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    add_up = (dctutil.idct_2d(theta_a.reshape(-1, hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), norm='ortho')).reshape(-1,hyp["PATCH_SIZE"] * hyp["PATCH_SIZE"], 1)
    result = torch.mul(src, mul_up) + add_up + delta
    return result

def train(hyp):

    if hyp["flag"] == "delta":
        from lfista.modelDelta import LFISTA
        phiWfun = phiW_delta
    elif hyp["flag"] == "additive":
        from lfista.modelAdditive import LFISTA
        phiWfun = phiW_additive
    elif hyp["flag"] == "full":
        from lfista.modelFull import LFISTA
        phiWfun = phiW_full

    sparseDim = hyp["sparseDim"]
    obsDim = hyp["obsDim"]

    batch_size = hyp["batch_size"]
    epochs = hyp["epochs"]
    info_period = hyp["info_period"]
    learning_rate = hyp["learning_rate"]

    device = hyp["device"]

    data_path = hyp["DATA_STORE_PATH"]
    output_path = hyp["modelOutputPath"]
    createDir(output_path)

    # # Create idct matrix
    # W = idct(np.eye(obsDim), axis = 0, norm='ortho')
    # W = torch.from_numpy(W).type(torch.float32).to(device)

    # dctutil = DCTUtility((hyp["PATCH_SIZE"], hyp["PATCH_SIZE"]), hyp["DCT_CUTOFF_FREQ"], gpu=2)
    # W = dctutil.getIDCTBases()

    # Load data
    # dataset_train = PatchDataset(hyp["DATA_STORE_PATH"], isTrain=True)
    # dataset_test = PatchDataset(hyp["DATA_STORE_PATH"], isTrain=False)

    ####################################################
    # Directly load all the data

    path = os.path.join(data_path, "train")
    patches_src_train = np.load(os.path.join(path, "src", "{}.npy".format(0)))
    patches_dst_train = np.load(os.path.join(path, "dst", "{}.npy".format(0)))

    patches_src_train = torch.Tensor(patches_src_train).type(torch.float32) / 255.
    patches_dst_train = torch.Tensor(patches_dst_train).type(torch.float32) / 255.

    dataset_train = TensorDataset(patches_src_train, patches_dst_train)

    # path = os.path.join(data_path, "test")
    # patches_src_test = np.load(os.path.join(path, "src", "{}.npy".format(0)))
    # patches_dst_test = np.load(os.path.join(path, "dst", "{}.npy".format(0)))

    # patches_src_test = torch.Tensor(patches_src_test).type(torch.float32) / 255.
    # patches_dst_test = torch.Tensor(patches_dst_test).type(torch.float32) / 255.

    # dataset_test = TensorDataset(patches_src_test, patches_dst_test)

    ####################################################

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True, shuffle=True)
    # dataloader_test = DataLoader(dataset_test, batch_size=batch_size, pin_memory=True, shuffle=True)

    epoch_start = 0
    net = LFISTA(hyp)
    epoch_start = 35
    # trained_model_path = os.path.join(hyp["modelOutputPath"], "model_epoch{}.pt".format(epoch_start))
    trained_model_path = os.path.join("lfista/trained/output_art_projection", "model_epoch{}.pt".format(epoch_start))
    
    # net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(trained_model_path, map_location=device))
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    loss_criterion = torch.nn.MSELoss()
    # loss_criterion = torch.nn.L1Loss()

    training_losses = []
    testing_losses = []
    
    plt.ion()
    print("Training Model:")
    
    for epoch in range(epoch_start, epochs):
        
        loss_all = 0
        print("Epoch {}:".format(epoch+1))

        net.train()
        prog_bar = tqdm(dataloader_train)
        for idx, (src, dst) in enumerate(prog_bar):

            src = src.to(device)
            dst = dst.to(device)

            # Set gradients of optimized tensors to zero
            optimizer.zero_grad()

            # Y = dst; Phi = src
            X_pred= net(src, dst)
            dst_pred = phiWfun(X_pred, src, hyp)
            loss = loss_criterion(dst, dst_pred)
            # loss = loss.mean()
            loss_all += float(loss.item())
            # Backward Propogation
            loss.backward()
            optimizer.step()
            
            # Following line was commented in working lista
            net.normalize() 
            # net.module.normalize()
            
            if idx % info_period == 0:
                prog_bar.set_description("Error {}".format(loss.item()))


            torch.cuda.empty_cache()

        

        avg_training_loss = loss_all / (idx + 1)
        training_losses.append(avg_training_loss)
        scheduler.step(avg_training_loss)
        
        
        print(torch.cat((dst[0], dst_pred[0]), dim=1))
        print("Number of non-zeros:", torch.count_nonzero(X_pred[0]))
        # print("Alpha:", net.alpha.data)
        
        # Run on test data
        # net.eval()
        # test_loss_all = 0
        # prog_bar = tqdm(dataloader_test)
        # with torch.no_grad():
        #     for idx, (src, dst) in enumerate(prog_bar):

        #         src = src.to(device)
        #         dst = dst.to(device)

        #         src = src.view(-1, obsDim, 1)
        #         dst = dst.view(-1, obsDim, 1)

        #         X_pred= net(src, dst)
        #         dst_pred = phiWfun(X_pred, src, hyp)
        #         loss_test = loss_criterion(dst, dst_pred)
        #         # loss_test = loss_test.mean()
        #         test_loss_all += float(loss_test.item())

        #         if idx % info_period == 0:
        #             prog_bar.set_description("Error {}".format(loss_test.item()))

        #         torch.cuda.empty_cache()

        # avg_testing_loss = test_loss_all / (idx + 1)
        # testing_losses.append(avg_testing_loss)
        
        # plot(np.linspace(0, len(training_losses), len(training_losses)), training_losses, testing_losses, savePath=hyp["modelOutputPath"], gui=False)
        torch.save(loss_all, output_path + "loss_epoch{}.pt".format(epoch+1))
        torch.save(net.state_dict(), output_path+"model_epoch{}.pt".format(epoch+1))

        print(
            "epoch [{}/{}], Training loss:{:.8f}, Testing loss:{:.8f} ".format(
                epoch + 1, hyp["epochs"], avg_training_loss, 0#avg_testing_loss
            )
        )

    return net
