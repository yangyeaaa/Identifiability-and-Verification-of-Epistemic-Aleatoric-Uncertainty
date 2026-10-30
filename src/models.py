"""
Models used in the paper.

  HeteroMCDropout  - the shared heteroscedastic network (Section III-A): head (mu, log var) trained by
                     Gaussian NLL. MC dropout: epistemic = variance of the mean across M passes, aleatoric =
                     mean predicted variance. K deterministic members give the deep-ensemble decomposition;
                     LaplaceDecomposition gives LA-1 / LA-2 on the same trained network.
  FreeHead         - free-output construction (Section II-B, Table I): mu, sigma_ale and sigma_epi are
                     separate bounded outputs with one learnable log-scale per dispersion head, trained by
                     NLL on the total variance.
  ClassicMCDropout - Gal & Ghahramani (2016) regression: point-output network with dropout at test time;
                     predictive variance = Var_i[mu_i] + sigma_n^2 with one noise level fitted on validation.
  QuantileNet      - multi-quantile head trained by the pinball loss.
"""
import numpy as np, torch, torch.nn as nn
from backbone import iTransformerEncoder, device, fit
import os
_EPOCHS = int(os.environ.get("UQ_EPOCHS", "50"))

ENC = dict(d_model=64, n_heads=4, n_layers=2)


def _enable_dropout(model):
    """MC-dropout sampling mode. The model has no BatchNorm, so train() only switches dropout on.
    Putting the whole model (not just nn.Dropout modules) in train mode matters: otherwise
    nn.TransformerEncoderLayer takes its inference fast path under no_grad and silently skips its
    internal dropout, so the sampled posterior differs from the one used in training."""
    model.train()


def rmse_val(model, x, y):
    mu = model.mean(x)
    return torch.sqrt(((y - mu) ** 2).mean())


# ======================================================================= proposed
class HeteroMCDropout(nn.Module):
    def __init__(self, seq_len, n_var, dropout=0.1, homoscedastic=False):
        super().__init__()
        d = ENC["d_model"]
        self.encoder = iTransformerEncoder(seq_len, n_var, dropout=dropout, **ENC)
        self.shared = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(d, 1)
        self.homoscedastic = homoscedastic
        if homoscedastic:
            self.logvar_global = nn.Parameter(torch.zeros(()))
        else:
            self.logvar_head = nn.Linear(d, 1)
            nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x):
        f = self.shared(self.encoder(x))
        mu = self.mu_head(f).squeeze(-1)
        if self.homoscedastic:
            logvar = self.logvar_global.expand_as(mu)
        else:
            logvar = torch.clamp(self.logvar_head(f).squeeze(-1), -7.0, 3.0)
        return mu, logvar

    def mean(self, x):
        return self.forward(x)[0]

    beta = 0.0   # beta-NLL (Seitzer et al., 2022): per-sample NLL weighted by stopgrad(sigma^(2 beta)); 0 = plain NLL

    @staticmethod
    def nll(model, x, y):
        mu, logvar = model(x)
        l = 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))
        if model.beta > 0:
            l = l * torch.exp(logvar).detach() ** model.beta
        return l.mean()

    def train_model(self, tl, vl, save, epochs=None, patience=12, lr=5e-4):
        epochs = epochs or _EPOCHS
        return fit(self, HeteroMCDropout.nll, tl, vl, epochs, patience, lr, save=save, val_fn=rmse_val)

    @torch.no_grad()
    def predict(self, X, M=40, mc_dropout=True, seed=None):
        """Returns normalized-unit arrays: mu, ale (std), epi (std), tot (std), and per-sample R_epi."""
        if seed is not None:
            torch.manual_seed(seed)
        if mc_dropout:
            _enable_dropout(self)
        else:
            self.eval(); M = 1
        mus, vars_ = [], []
        for _ in range(M):
            mu, logvar = self(X.to(device))
            mus.append(mu.cpu().numpy()); vars_.append(np.exp(logvar.cpu().numpy()))
        mus = np.stack(mus); vars_ = np.stack(vars_)
        var_ale = vars_.mean(0); var_epi = mus.var(0, ddof=1) if M > 1 else np.zeros_like(var_ale)   # unbiased (Bessel)
        return _pack(mus.mean(0), var_ale, var_epi)


def _pack(mu, var_ale, var_epi):
    var_tot = var_ale + var_epi
    return dict(mu=mu, ale=np.sqrt(var_ale), epi=np.sqrt(var_epi), tot=np.sqrt(var_tot),
                r_epi=var_epi / (var_tot + 1e-12), R_epi=float(np.mean(var_epi / (var_tot + 1e-12))))


@torch.no_grad()
def ensemble_predict(members, X):
    """Deep-ensemble decomposition (law of total variance over K deterministic members)."""
    mus, vars_ = [], []
    for m in members:
        m.eval(); mu, logvar = m(X.to(device))
        mus.append(mu.cpu().numpy()); vars_.append(np.exp(logvar.cpu().numpy()))
    mus = np.stack(mus); vars_ = np.stack(vars_)
    return _pack(mus.mean(0), vars_.mean(0), mus.var(0, ddof=1))   # unbiased (Bessel) between-member variance


# ======================================================================= free-head construction
class FreeHead(nn.Module):
    """Eq. (freehead): sigma_ale and sigma_epi are separate outputs, each a sigmoid bounded in
    [sigma_min, sigma_max] times exp(log_scale); the likelihood sees only their sum of squares."""
    def __init__(self, seq_len, n_var, dropout=0.1, init_ale=0.0, init_epi=-1.0,
                 sigma_min=0.01, sigma_max=3.0):
        super().__init__()
        d = ENC["d_model"]
        self.encoder = iTransformerEncoder(seq_len, n_var, dropout=dropout, **ENC)
        self.shared = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(d, 1)
        self.ale_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.epi_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.log_ale_scale = nn.Parameter(torch.tensor(float(init_ale)))
        self.log_epi_scale = nn.Parameter(torch.tensor(float(init_epi)))
        self.smin, self.smax = sigma_min, sigma_max

    def forward(self, x):
        f = self.shared(self.encoder(x))
        mu = self.mu_head(f).squeeze(-1)
        b = lambda raw: torch.sigmoid(raw.squeeze(-1)) * (self.smax - self.smin) + self.smin
        s_ale = b(self.ale_head(f)) * torch.exp(self.log_ale_scale)
        s_epi = b(self.epi_head(f)) * torch.exp(self.log_epi_scale)
        return mu, s_ale, s_epi

    def mean(self, x):
        return self.forward(x)[0]

    @staticmethod
    def nll(model, x, y):
        mu, s_ale, s_epi = model(x)
        var = s_ale ** 2 + s_epi ** 2 + 1e-6
        return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()

    def train_model(self, tl, vl, save, epochs=None, patience=12, lr=5e-4):
        epochs = epochs or _EPOCHS
        return fit(self, FreeHead.nll, tl, vl, epochs, patience, lr, save=save, val_fn=rmse_val)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        mu, s_ale, s_epi = self(X.to(device))
        return _pack(mu.cpu().numpy(), s_ale.cpu().numpy() ** 2, s_epi.cpu().numpy() ** 2)


# ======================================================================= classic MC dropout
class ClassicMCDropout(nn.Module):
    """Point-output network; uncertainty only from dropout at test time plus a homoscedastic
    noise term estimated on the validation partition (Gal & Ghahramani, 2016)."""
    def __init__(self, seq_len, n_var, dropout=0.1):
        super().__init__()
        d = ENC["d_model"]
        self.encoder = iTransformerEncoder(seq_len, n_var, dropout=dropout, **ENC)
        self.shared = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.mu_head = nn.Linear(d, 1)
        self.noise_var = 0.0

    def forward(self, x):
        return self.mu_head(self.shared(self.encoder(x))).squeeze(-1)

    def mean(self, x):
        return self.forward(x)

    @staticmethod
    def mse(model, x, y):
        return ((y - model(x)) ** 2).mean()

    def train_model(self, tl, vl, save, epochs=None, patience=12, lr=5e-4):
        epochs = epochs or _EPOCHS
        return fit(self, ClassicMCDropout.mse, tl, vl, epochs, patience, lr, save=save, val_fn=rmse_val)

    @torch.no_grad()
    def _passes(self, X, M, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        _enable_dropout(self)
        return np.stack([self(X.to(device)).cpu().numpy() for _ in range(M)])

    def fit_noise(self, Xv, yv, M=40):
        mus = self._passes(Xv, M, seed=7)
        resid_var = np.mean((yv - mus.mean(0)) ** 2)
        self.noise_var = float(max(resid_var - mus.var(0, ddof=1).mean(), 1e-6))

    @torch.no_grad()
    def predict(self, X, M=40, seed=None):
        mus = self._passes(X, M, seed)
        return _pack(mus.mean(0), np.full(mus.shape[1], self.noise_var), mus.var(0, ddof=1))


# ======================================================================= quantile regression
QGRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


class QuantileNet(nn.Module):
    def __init__(self, seq_len, n_var, dropout=0.1):
        super().__init__()
        self.encoder = iTransformerEncoder(seq_len, n_var, dropout=dropout, **ENC)
        self.head = nn.Linear(ENC["d_model"], len(QGRID))
        self.register_buffer("q", torch.tensor(QGRID, dtype=torch.float32))

    def forward(self, x):
        return self.head(self.encoder(x))

    @staticmethod
    def pinball(model, x, y):
        pred = model(x); e = y.unsqueeze(1) - pred; q = model.q.unsqueeze(0)
        return torch.maximum(q * e, (q - 1) * e).mean()

    def train_model(self, tl, vl, save, epochs=None, patience=12, lr=5e-4):
        epochs = epochs or _EPOCHS
        return fit(self, QuantileNet.pinball, tl, vl, epochs, patience, lr, save=save)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        Q = self(X.to(device)).cpu().numpy()
        return np.sort(Q, axis=1)


def new_model(kind, sc, **kw):
    cls = dict(ours=HeteroMCDropout, free=FreeHead, classic=ClassicMCDropout, qr=QuantileNet)[kind]
    return cls(sc.sl, sc.nv, **kw)


# ======================================================================= Laplace decompositions
class LaplaceDecomposition:
    """Post-hoc Laplace (GGN) posterior over the last one or two layers of a trained HeteroMCDropout
    network, with the prior precision set by evidence maximisation (MacKay fixed point).

      layers=1 : mean head (64 weights + bias)
      layers=2 : shared hidden layer (64x64 + 64) and mean head  (4225 parameters)

    The predictive mean is the deterministic network output; the aleatoric variance is the
    heteroscedastic head; the epistemic variance is J(x)^T Sigma J(x), J = d mu / d theta.
    The Hessian is weighted by the predicted aleatoric precision (heteroscedastic GGN)."""

    def __init__(self, layers=1):
        self.layers = layers

    @torch.no_grad()
    def _parts(self, m, X):
        m.eval()
        h = m.encoder(X.to(device)); z = m.shared[0](h); f = m.shared(h)
        mu = m.mu_head(f).squeeze(-1); lv = torch.clamp(m.logvar_head(f).squeeze(-1), -7.0, 3.0)
        return h.double().numpy(), z.double().numpy(), f.double().numpy(), mu.double().numpy(), np.exp(lv.double().numpy())

    def _jac(self, m, h, z, f):
        from scipy.stats import norm as _n
        ones = np.ones((len(f), 1))
        if self.layers == 1:
            return np.concatenate([f, ones], 1)
        w = m.mu_head.weight.detach().double().numpy().ravel()
        gz = w[None] * (_n.cdf(z) + z * _n.pdf(z))                       # exact GELU derivative
        JW1 = (gz[:, :, None] * h[:, None, :]).reshape(len(h), -1)
        return np.concatenate([JW1, gz, f, ones], 1)

    def _theta(self, m):
        w = [m.mu_head.weight.detach().double().numpy().ravel(), m.mu_head.bias.detach().double().numpy()]
        if self.layers == 2:
            lin = m.shared[0]
            w = [lin.weight.detach().double().numpy().ravel(), lin.bias.detach().double().numpy()] + w
        return np.concatenate(w)

    def fit(self, m, Xtr, chunk=1024):
        H = None
        for i in range(0, len(Xtr), chunk):
            h, z, f, _, s2 = self._parts(m, Xtr[i:i + chunk]); J = self._jac(m, h, z, f)
            Hc = (J / s2[:, None]).T @ J; H = Hc if H is None else H + Hc
        th = self._theta(m); lam = np.clip(np.linalg.eigvalsh(H), 0, None); a = 1.0
        for _ in range(1000):
            g = np.sum(lam / (lam + a)); a_new = g / (th @ th)
            if abs(a_new - a) < 1e-10 * a:
                break
            a = a_new
        self.alpha, self.gamma, self.n_params = a, g, len(th)
        self.Sigma = np.linalg.inv(H + a * np.eye(len(th)))
        return self

    def predict(self, m, X, chunk=1024):
        mus, s2s, ves = [], [], []
        for i in range(0, len(X), chunk):
            h, z, f, mu, s2 = self._parts(m, X[i:i + chunk]); J = self._jac(m, h, z, f)
            ves.append(np.sum((J @ self.Sigma) * J, axis=1)); mus.append(mu); s2s.append(s2)
        return _pack(np.concatenate(mus), np.concatenate(s2s), np.concatenate(ves))
