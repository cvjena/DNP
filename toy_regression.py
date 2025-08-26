import numpy as np
from dnp_model import RegressionDNP
from dataset import toy_regression_dataset
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam
from scipy.signal import savgol_filter
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

def visualize(model, dx, dy, stdx, stdy, cond_x=None, cond_y=None, all_x=None, all_y=None, samples=30, 
              range_y=(-100., 100.), title='', train=False):
    '''
    Visualizes the predictive distribution
    '''
    dxy = np.zeros((dx.shape[0], samples))
    if not train:
        model.eval()
    with torch.no_grad():
        dxi = torch.from_numpy(stdx.transform(dx).astype(np.float32))
        if torch.cuda.is_available():
            dxi = dxi.cuda()
        for j in range(samples):
            dxy[:, j] = model.predict(dxi.unsqueeze(0), cond_x.unsqueeze(0), cond_y.unsqueeze(0)).ravel()
    print()

    plt.figure()
    plt.tick_params(axis='both', labelsize=20)
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)
    
    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = all_x.cpu(), all_y.cpu(), cond_x.cpu(), cond_y.cpu()

    plt.plot(dx.ravel(), mean_dxys, label='Mean function')
    plt.plot(dx.ravel(), dy, label='True function')
    plt.plot(stdx.inverse_transform(cond_x.data.numpy()).ravel(), stdy.inverse_transform(cond_y.data.numpy()).ravel(), 'o',
             label='Observations')

    plt.fill_between(dx.ravel(), mean_dxys-1.*std_dxys, mean_dxys+1.*std_dxys, color='indigo', alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-2.*std_dxys, mean_dxys+2.*std_dxys, color='indigo', alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-3.*std_dxys, mean_dxys+3.*std_dxys, color='indigo', alpha=.1)

    plt.xlim([np.min(dx), np.max(dx)])
    plt.ylim([-3,3])

    model.train()
    plt.show()



X, y, dx, dy = toy_regression_dataset()

stdx, stdy = StandardScaler().fit(X), StandardScaler().fit(y)
X, y = stdx.transform(X), stdy.transform(y)
idx = np.arange(X.shape[0])
idxC = np.random.choice(idx, size=(5,), replace=False)
idxT = np.array([i for i in idx if i not in idxC.tolist()])

XC, yC = torch.from_numpy(X[idxC].astype(np.float32)), torch.from_numpy(y[idxC].astype(np.float32))
XT, yT = torch.from_numpy(X[idxT].astype(np.float32)), torch.from_numpy(y[idxT].astype(np.float32))
X, y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

torch.manual_seed(0)

dnp = RegressionDNP(dim_x=1, dim_y=1, transf_y=stdy, dim_h=64, dim_u=64, n_layers=1, dim_z=64, fb_z=1.0, lambda_min=0.1, lambda_max=1.0, beta=1.0)

if torch.cuda.is_available():
    XC, XT, X = XC.cuda(), XT.cuda(), X.cuda()
    yC, yT, y = yC.cuda(), yT.cuda(), y.cuda()
    dnp = dnp.cuda()

optimizer = Adam(dnp.parameters(), lr=1e-3)
dnp.train()

epochs = 650
for i in range(epochs):
    optimizer.zero_grad()
    loss = dnp(XC.unsqueeze(0), yC.unsqueeze(0), XT.unsqueeze(0), yT.unsqueeze(0))
    loss.backward()
    optimizer.step()
        
    if i % int(epochs / 5) == 0:
        print('Epoch {}/{}, loss: {:.3f}'.format(i, epochs, loss.item()))
        visualize(dnp, dx, dy, stdx, stdy, cond_x=XC, cond_y=yC, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)
visualize(dnp, dx, dy, stdx, stdy, cond_x=XC, cond_y=yC, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)
print('Done.')