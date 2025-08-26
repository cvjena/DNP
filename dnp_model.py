import torch.nn as nn
import torch
import torch.nn.functional as F
from dnp_utils import Normal
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.utils.parametrizations import spectral_norm



def singular_value_constraint_loss_svd(matrix, lambda_min=0.1, lambda_max=1.0):
    """
    Computes a loss that penalizes:
      - the maximum singular value being greater than lambda1, and
      - the minimum singular value being less than lambda2.
    
    Args:
        matrix (Tensor): A 2D tensor of shape (M, N) or a batched tensor of shape (B, M, N).
        lambda1 (float): The upper bound for the maximum singular value.
        lambda2 (float): The lower bound for the minimum singular value.
    
    Returns:
        loss (Tensor): A scalar loss value.
    """
    U, S, V = torch.linalg.svd(matrix, full_matrices=False)
    
    # S is 1D for a single matrix (or 2D for batched matrices)
    if S.dim() == 2:
        sigma_max = S.max(dim=-1)[0]
        sigma_min = S.min(dim=-1)[0]
    else:
        sigma_max = S[0]
        sigma_min = S[-1]
    
    loss = ((F.relu(sigma_max - lambda_max))**2 + (F.relu(lambda_min - sigma_min))**2).mean()
    return loss


def singular_value_constraint_loss_lobpcg(matrix, lambda_min, lambda_max, k=1, max_iter=20, tol=1e-5):
    """
    Approximates singular values using LOBPCG and computes a constraint loss.
    
    Args:
        matrix (Tensor): A 2D tensor of shape (M, N). Batched input not yet supported here.
        lambda1 (float): The upper bound for the maximum singular value.
        lambda2 (float): The lower bound for the minimum singular value.
        k (int): Number of eigenvalues to approximate (should be 1 for largest/smallest).
        max_iter (int): Maximum number of LOBPCG iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        loss (Tensor): A scalar loss value.
    """
    A = matrix
    U, S, V = torch.linalg.svd(matrix, full_matrices=False)
   
    #if matrix.dim() == 2:
        # Form the symmetric positive semi-definite matrix
    #    WWT = matrix @ matrix.T
    #else:
    #    WWT = matrix @ matrix.transpose(1, 2)  # Symmetric PSD
    M, N = matrix.shape
    min_dim = min(M, N)
    if min_dim < 3 * k:
        sigma_max = S.max(dim=-1)[0]
        sigma_min = S.min(dim=-1)[0]

    else:
        if M < N:
            WWT = matrix @ matrix.T
        else:
            WWT = matrix.T @ matrix 

        with torch.no_grad():
            # Approximate the largest eigenvalue
            eigvals_max, _ = torch.lobpcg(WWT, k=k, method="ortho", largest=True, niter=max_iter, tol=tol)
            sigma_max = eigvals_max.sqrt().squeeze()

            # Approximate the smallest eigenvalue
            eigvals_min, _ = torch.lobpcg(WWT, k=k, method="ortho", largest=False, niter=max_iter, tol=tol)
            sigma_min = eigvals_min.sqrt().squeeze()

        # Compute the regularization loss
    loss = (F.relu(sigma_max - lambda_max)**2 + F.relu(lambda_min - sigma_min)**2)
    return loss


def mlp_singular_value_loss(model, lambda_min, lambda_max):
    """
    Applies the singular value constraint loss to the weight matrices of all nn.Linear layers in the model.
    
    Args:
        model (nn.Module): The MLP model.
        lambda1 (float): Maximum singular value threshold.
        lambda2 (float): Minimum singular value threshold.
    
    Returns:
        total_loss (Tensor): The summed constraint loss over all linear layers.
    """
    total_loss = 0.0
    num_layers = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # module.weight has shape (out_features, in_features)
            total_loss += singular_value_constraint_loss_lobpcg(module.weight, lambda_min, lambda_max)
            num_layers += 1
    # Optionally average over the number of layers
    if num_layers > 0:
        total_loss /= num_layers
    return total_loss



class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class ConvEncoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(ConvEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(x_dim, 16, kernel_size=3, padding=1),  # Output: [batch, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                            # Output: [batch, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: [batch, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                             # Output: [batch, 32, 7, 7]
        )

        self.input_to_hidden = nn.Sequential(*self.conv_layers)
        #self.fc = nn.Linear(1568 + y_dim, r_dim)  #Mnist
        self.fc = nn.Linear(2048 + y_dim, r_dim)
        #self.conv_layers = LeNet()
        #self.conv_layers = densenet_cifar100()
        #self.conv_layers = VGGCustom()
        #self.fc = nn.Linear(500 + y_dim, r_dim)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        conv_out = self.conv_layers(x)           # shape: [batch, 32, 7, 7]
        conv_out = conv_out.view(x.size(0), -1)    # flatten to [batch, 1568]
        
        # Concatenate the image features with the one-hot label
        concat = torch.cat([conv_out, y], dim=-1)  # shape: [batch, 1568 + 10]
        
        # Pass through a fully connected layer to get the latent representation
        latent = self.fc(concat)  # shape: [batch, embedding_dim]
        return latent

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv):
        super(CrossAttention, self).__init__()
        self.scale = (dim_q) ** 0.5  

    def forward(self, Q, K):

        #attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (batch, 10, 20)
        attention_scores = -torch.cdist(Q, K, p=2) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize across 20 points


        return attention_weights 
    

class RegressionDNP(nn.Module):
    """
    Distance-informed Neural Process for regression
    """
    def __init__(self, dim_x=1, dim_y=1, dim_h=50, transf_y=None, n_layers=1, dim_u=1, dim_z=1, fb_z=0.,
                 lambda_min=0.1, lambda_max=1.0, beta=1.0):
        '''
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        '''
        super(RegressionDNP, self).__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.beta = beta

        self.xy_to_r = Encoder(dim_x, dim_y, dim_h, dim_h)
        self.r_to_mu_sigma = MuSigmaEncoder(dim_h, dim_z)

        self.register_buffer('lambda_z', torch.tensor(1e-8)) 

        # transformation of the input
        init = [nn.Linear(dim_x, self.dim_h), nn.LeakyReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.LeakyReLU()]

        global_init = [nn.Linear(dim_x+dim_y, self.dim_h), nn.LeakyReLU()]
        for i in range(n_layers - 1):
            global_init += [nn.Linear(self.dim_h, self.dim_h), nn.LeakyReLU()]

        self.cond_trans = nn.Sequential(*init)

        self.global_cond_trans = nn.Sequential(*global_init)
        self.p_u = nn.Linear(self.dim_h, 1 * self.dim_u)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)
        self.p_zg = nn.Linear(self.dim_h, 2 * self.dim_z)

        # Self attention between the latent embeddings of reference set
        self.cross_attention = CrossAttention(dim_u, dim_u)

        # p(y|z) or p(y|z, u)
        self.output = nn.Sequential(nn.Linear(self.dim_z + self.dim_u + self.dim_z, self.dim_h),
                                    nn.LeakyReLU(), nn.Linear(self.dim_h, 2 * dim_y))

    
    def xy_to_mu_sigma(self, x, y):
        
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """   
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.dim_x)
        y_flat = y.contiguous().view(batch_size * num_points, self.dim_y)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.dim_h)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)      
        """
        return torch.mean(r_i, dim=1)
    
    def forward(self, XC, yC, XT, yT, kl_anneal=1.):
        
        X_all = torch.cat([XC, XT], dim=1)   #target set
        y_all = torch.cat([yC, yT], dim=1)   #target set

        H_all = self.cond_trans(X_all)   #target set

        u = self.p_u(H_all)

        # get A
        A_t_attention_probs = self.cross_attention(u[:, XC.size(1):], u[:, 0:XC.size(1)])  #context+target set
        A_t = A_t_attention_probs.reshape(-1, XT.size(1), XC.size(1))

        A_c_attention_probs = self.cross_attention(u[:, 0:XC.size(1)], u[:, 0:XC.size(1)])  #context+target set
        A_c = A_c_attention_probs.reshape(-1, XC.size(1), XC.size(1))
        
        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(torch.cat([H_all], dim=-1)), self.dim_z, dim=-1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z_t = qz.rsample()
        qz_mean_context, qz_logscale_context = qz_mean_all[:, 0:XC.size(1)], qz_logscale_all[:, 0:XC.size(1)]
        qz_mean_target, qz_logscale_target = qz_mean_all[:, XC.size(1):], qz_logscale_all[:, XC.size(1):]
        qz_target = Normal(qz_mean_target, qz_logscale_target)
        #z_t = qz_target.rsample()

        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(yC), self.dim_z, dim=-1)

        pz_mean_all = torch.matmul(torch.cat([A_c, A_t], dim=1), cond_y_mean + qz_mean_context)
        pz_logscale_all = torch.matmul(torch.cat([A_c, A_t], dim=1), cond_y_logscale + qz_logscale_context)

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z_t) - qz.log_prob(z_t)

        #context_data = torch.cat([XR, yR], dim=-1)
        #r = self.global_cond_trans(context_data)
        #pz_global_mu, pz_global_sigma = torch.split(self.p_zg(r), self.dim_z, dim=-1)
        pz_global_mu_c, pz_global_sigma_c = self.xy_to_mu_sigma(XC, yC)
        pz_global_c = Normal(pz_global_mu_c, pz_global_sigma_c)
        z_global_c = pz_global_c.rsample()

        pz_global_mu_t, pz_global_sigma_t = self.xy_to_mu_sigma(XT, yT)
        pz_global_t = Normal(pz_global_mu_t, pz_global_sigma_t)
        z_global_t = pz_global_t.rsample()


        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = - torch.sum(pqz_all, dim=list(range(1, pqz_all.ndim)))

            #self.lambda_z = self.lambda_z.expand(batch_size, 1)
            

            if self.training:
                # Compute per-batch thresholds
                upper_threshold = self.fb_z * z_t.size(1) * z_t.size(2) * (1 + 0.05)
                lower_threshold = self.fb_z * z_t.size(1) * z_t.size(2)

                # Adjust lambda_z per batch
                increase_mask = log_qpz > upper_threshold
                decrease_mask = log_qpz < lower_threshold

                # Update lambda_z per batch in a vectorized way
                self.lambda_z = torch.clamp(
                    self.lambda_z * (1 + 0.1 * increase_mask.float() - 0.1 * decrease_mask.float()),
                    min=1e-8, max=1.
                )
                self.lambda_z = self.lambda_z.mean()

            batch_size_XC = XC.size(1)

    
            log_pqz_C = self.lambda_z * torch.sum(pqz_all[:, :batch_size_XC], dim=list(range(1, pqz_all.ndim)))
            log_pqz_T = self.lambda_z * torch.sum(pqz_all[:, batch_size_XC:], dim=list(range(1, pqz_all.ndim)))

        else:
            log_pqz_C = torch.sum(pqz_all[0:XC.size(1)], dim=list(range(1, pqz_all.ndim)))
            log_pqz_T = torch.sum(pqz_all[XC.size(1):], dim=list(range(1, pqz_all.ndim)))


        batch_size, num_points, _ = z_t.size()
        z_global_c = z_global_c.unsqueeze(1).repeat(1, num_points, 1)
        final_rep = torch.cat([z_t, u, z_global_c], dim=-1)
        mean_y, logstd_y = torch.split(self.output(final_rep), self.dim_y, dim=-1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        mean_yC, mean_yT = mean_y[:, :XC.size(1)], mean_y[:, XC.size(1):]
        logstd_yC, logstd_yT = logstd_y[:, 0:XC.size(1)], logstd_y[:, XC.size(1):]

        # logp(R)
        pyC = Normal(mean_yC, logstd_yC)
        log_pyC = torch.sum(pyC.log_prob(yC))

        # logp(M|S)

        pyT = Normal(mean_yT, logstd_yT)
        log_pyT = torch.sum(pyT.log_prob(yT))

        obj_C = (log_pyC + log_pqz_C) / float(XC.size(0))
        obj_T = (log_pyT + log_pqz_T) / float(XT.size(0))

        q_target = torch.distributions.Normal(pz_global_mu_t, pz_global_sigma_t)
        q_context = torch.distributions.Normal(pz_global_mu_c, pz_global_sigma_c)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()

        spectral_loss = mlp_singular_value_loss(self.cond_trans, self.lambda_min, self.lambda_max)

        #print(spectral_loss)
        obj = obj_T + obj_C 
        obj = torch.mean(obj) 

        loss = - obj + kl + self.beta * spectral_loss

        return loss

    def predict(self, x_new, XC, yC, cov=False, sample=True):

        H_all = self.cond_trans(torch.cat([XC, x_new], dim=1))
        # get U
        #pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=-1)
        #pu = Normal(pu_mean_all, pu_logscale_all)
        #u = pu.rsample()
        u = self.p_u(H_all)
        #A = sample_bipartite(u[XR.size(0):], u[0:XR.size(0)], self.pairwise_g, training=False)
        A_attention_probs = self.cross_attention(u[:, XC.size(1):], u[:, 0:XC.size(1)])
        A = A_attention_probs.reshape(-1, x_new.size(1), XC.size(1))
        #A = sample_bipartite(u[:, XR.size(1):], u[:, 0:XR.size(1)], self.pairwise_g, training=self.training)
        pz_mean_all, pz_logscale_all = torch.split(self.q_z(torch.cat([H_all[:, 0:XC.size(1)]], dim=-1)), self.dim_z, -1)
        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(yC), self.dim_z, -1)

        pz_mean_all = torch.matmul(A, cond_y_mean + pz_mean_all)
        pz_logscale_all = torch.matmul(A, cond_y_logscale + pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()

        pz_global_mu, pz_global_sigma = self.xy_to_mu_sigma(XC, yC)
        pz_global = Normal(pz_global_mu, pz_global_sigma)
        z_global = pz_global.rsample()

        batch_size, num_points, _ = z.size()
        z_global = z_global.unsqueeze(1).repeat(1, num_points, 1)

        final_rep = torch.cat([z, u[:, XC.size(1):], z_global], dim=-1)
        mean_y, logstd_y = torch.split(self.output(final_rep), self.dim_y, dim=-1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            y_pred = y_pred.squeeze(0)
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        if cov:
            return torch.distributions.Normal(mean_y, logstd_y.exp())
        else:
            return y_pred
