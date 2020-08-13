import numpy as np
import torch
import torch.nn as nn

from .helper_module import Encoder, Decoder


class CVAE(nn.Module):
    """
            CVAE class. This class contains the implementation Conditional
            Variational Auto-encoder.
            # Parameters

                input_dim: integer
                    Number of input features (i.e. gene in case of scRNA-seq).
                num_classes: integer
                    Number of classes (conditions) the data contain. if `None` the model
                    will be a normal VAE instead of conditional VAE.
                encoder_layer_sizes: List
                    A list of hidden layer sizes for encoder network.
                latent_dim: integer
                    Bottleneck layer (z)  size.
                decoder_layer_sizes: List
                    A list of hidden layer sizes for decoder network.
                alpha: float
                     alpha coefficient for KL loss.
                use_batch_norm: boolean
                    if `True` batch normalization will applied to hidden layers
                dr_rate: float
                    Dropput rate applied to hidden layer, if `dr_rate`==0 no dropput will be applied.
                use_mmd: boolean
                    if `True` then MMD will be applied to first decoder layer.
                beta: float
                    beta coefficient for MMD loss.

        """

    def __init__(self, input_dim, num_classes=None, encoder_layer_sizes=[64, 32],
                 latent_dim=10, decoder_layer_sizes=[32, 64], alpha=0.001, use_batch_norm=True,
                 dr_rate=0.2, use_mmd=True, beta=1, output_activation="ReLU"):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_dim) == int
        assert type(decoder_layer_sizes) == list
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_cls = num_classes
        self.use_mmd = use_mmd
        self.beta = beta
        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False
        self.use_bn = use_batch_norm
        self.alpha = alpha
        self.op_activation = output_activation
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes.append(self.input_dim)
        self.encoder = Encoder(encoder_layer_sizes, self.latent_dim,
                               self.use_bn, self.use_dr, self.dr_rate, self.num_cls)
        self.decoder = Decoder(decoder_layer_sizes, self.latent_dim,
                               self.use_bn, self.use_dr, self.dr_rate, self.use_mmd, self.num_cls, self.op_activation)

    def inference(self, n=1, c=None):
        """
                Generate `n` datapoints by sampling from a standard Gaussian and feeding them
                  to decoder.

                # Parameters
                    n: integer
                    c: `numpy nd-array`
                        `numpy nd-array` of original desired labels for each sample.
                # Returns
                    rec_data: 'numpy nd-array'
                        Returns 'numpy nd-array` containing reconstructed 'data' in shape [n, input_dim].
                """
        batch_size = n
        z = torch.randn([batch_size, self.latent_dim])
        if c is not None:
            c = torch.tensor(c)
        recon_x = self.decoder(z, c)
        return recon_x

    def sampling(self, mu, log_var):
        """
               Samples from standard Normal distribution with shape and
               applies reparametrization trick.
               # Parameters
                   mu: `Tensor`
                        mean vector
                   log_var: `Tensor`
                        log_var tensor
               # Returns
                   The computed Tensor of samples.
           """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def get_latent(self, x, c=None, mean=False):
        """
                  Map `x` in to the latent space. This function will feed data
                  in encoder  and return  z for each sample in data.
                  # Parameters
                      x:  numpy nd-array
                          Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                      c: `numpy nd-array`
                        `numpy nd-array` of original desired labels for each sample.
                      mean: boolean
                           if `True` only return mean vector ohterwise z will be fed to sampling function.
                  # Returns
                      latent: `numpy nd-array`
                          Returns array containing latent space encoding of 'x'
              """
        if c is not None:
            c = torch.tensor(c).to(self.device)
        x = torch.tensor(x).to(self.device)
        z_mean, z_log_var = self.encoder(x, c)
        z_sample = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean.cpu().data.numpy()
        return z_sample.cpu().data.numpy()

    def get_y(self, x, c=None):
        """
                       Map `x` in to the y dimension as described here https://arxiv.org/abs/1910.01791.
                        This function will feed data in encoder  and return  z for each sample in data.
                       # Parameters
                           x:  numpy nd-array
                               Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                           c: `numpy nd-array`
                             `numpy nd-array` of original desired labels for each sample.
                           mean: boolean
                                if `True` only return mean vector ohterwise z will be fed to sampling function.
                       # Returns
                           latent: `numpy nd-array`
                               Returns array containing latent space encoding of 'x'
                   """
        if c is not None:
            c = torch.tensor(c).to(self.device)
        x = torch.tensor(x).to(self.device)
        z_mean, z_log_var = self.encoder(x, c)
        z_sample = self.sampling(z_mean, z_log_var)
        _, y = self.decoder(z_sample, c)
        return y.cpu().data.numpy()

    def predict(self, x, y, target):
        """
                Predicts how data points `x` with original condition (classes) `y` will look like in `target` condition.
                # Parameters
                    x: `numpy nd-array`
                        nummpy data matrix containing source data points.
                    y: `numpy nd-array`
                        `numpy nd-array` of original labels .
                    target: str
                        target condition for the predcition.
                # Returns
                    output: `numpy nd-array`
                        `numpy nd-array`  of predicted cells in target condition.
                # Example
                ```python
                adata = sc.read("./data/kang_seurat.h5ad")
                sc.pp.normalize_per_cell(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=1000)
                adata = adata[:, adata.var['highly_variable']]
                n_conditions = adata.obs["condition"].unique().shape[0]
                adata_train = adata[~((adata.obs["cell_type"] == "pDC")
                                      & (adata.obs["condition"] == "CTRL"))]
                model = CVAE(adata_train.n_vars, num_classes=n_conditions,
                             encoder_layer_sizes=[64], decoder_layer_sizes=[64], latent_dim=10, alpha=0.0001,
                             use_mmd=True, beta=10)
                trainer = Trainer(model, adata_train)
                trainer.train_trvae(100, 64)
                data = model.get_y(adata.X.A, model.label_encoder.transform(adata.obs["condition"]))
                adata_latent = sc.AnnData(adata_train)
                adata_latent.obs["cell_type"] = adata_train.obs["cell_type"].tolist()
                adata_latent.obs["condition"] = adata_train.obs["condition"].tolist()
                sc.pp.neighbors(adata_latent)
                sc.tl.umap(adata_latent)
                sc.pl.umap(adata_latent, color=["condition", "cell_type"])
                ground_truth = adata_source = adata[(adata.obs["cell_type"] == "pDC")]
                adata_source = adata[(adata.obs["cell_type"] == "pDC") & (adata.obs["condition"] == "CTRL")]
                predicted_data = model.predict(x=adata_source.X.A, y=adata_source.obs["condition"].tolist(),
                                               target="STIM")
                ```
        """

        y = self.label_encoder.transform(np.array(y))
        z = self.get_latent(x, y)
        target_labels = np.array([target])
        target_labels = self.label_encoder.transform(np.tile(target_labels, len(y)))
        predicted = self.reconstruct(z, target_labels, use_latent=True)
        return predicted

    def reconstruct(self, x, c=None, use_latent=False):
        """
        Reconstruct the latent space encoding via the decoder.
        # Parameters
            x: `numpy nd-array`
                nummpy data matrix containing data points.
            c: `numpy nd-array`
                `numpy nd-array` of original labels. Only set in `None` for VAE
                model.
            use_latent: bool
                This flag determines whether the `x` is already in latent space or not.
                if `True`: The `x` is in latent space `x` is in shape [n_obs, latent_dim]).
                if `False`: The `x` is not in latent space (`data.X` is in shape [n_obs, input_dim]).
        # Returns
            rec_data: 'numpy nd-array'
                Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, input_dim].
        """
        if use_latent:
            x = torch.tensor(x).to(self.device)
            if c is not None:
                c = torch.tensor(c).to(self.device)
            if self.use_mmd:
                reconstructed, _ = self.decoder(x, c)
            else:
                reconstructed = self.decoder(x, c)
            return reconstructed.cpu().data.numpy()
        else:
            z = self.get_latent(x, c)
            z = torch.tensor(z).to(self.device)
            if c is not None:
                c = torch.tensor(c).to(self.device)
            if self.use_mmd:
                reconstructed, _ = self.decoder(z, c)
                return reconstructed
            else:
                reconstructed = self.decoder(z, c)
                return reconstructed.cpu().data.numpy()

    def forward(self, x, c=None):
        z_mean, z_log_var = self.encoder(x, c)
        z = self.sampling(z_mean, z_log_var)
        if self.use_mmd:
            recon_x, y = self.decoder(z, c)
            return recon_x, z_mean, z_log_var, y
        else:
            recon_x = self.decoder(z, c)
            return recon_x, z_mean, z_log_var
