import math
import torch


class DiffusionModel(torch.nn.Module):

    def __init__(
            self,
            noise_schedule: str,
            noise_scale: float,
            min_noise: float,
            max_noise: float,
            max_diffusion_steps: int,
            embedding_dim: int,
            device: str = 'cuda'
    ):
        super(DiffusionModel, self).__init__()

        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.max_diffusion_steps = max_diffusion_steps
        self.embedding_dim = embedding_dim
        self.device = device

        # Calculate betas
        self.betas = self.calculate_betas()

        # Initialize diffusion model settings
        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=device)], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0).to(device)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def _betas_from_linear_variance(self, variance: torch.Tensor, max_beta: float = 0.999) -> torch.Tensor:
        alpha_bar = 1 - variance
        betas = [1 - alpha_bar[0]]
        for i in range(1, self.max_diffusion_steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return torch.tensor(betas, device=self.device, dtype=torch.float32)

    def _betas_for_alpha_bar(self, max_beta: float = 0.999) -> torch.Tensor:
        betas = []
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        for i in range(self.max_diffusion_steps):
            t1 = i / self.max_diffusion_steps
            t2 = (i + 1) / self.max_diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, device=self.device, dtype=torch.float32)

    def calculate_betas(self) -> torch.Tensor:
        """
        Calculate betas. noise_schedule can be 'linear', 'linear-var', 'cosine', or 'binomial'
        :return:
        """
        if self.noise_schedule in ['linear', 'linear-var']:
            min_betas = self.noise_scale * self.min_noise
            max_betas = self.noise_scale * self.max_noise
            betas = torch.linspace(
                start=min_betas,
                end=max_betas,
                steps=self.max_diffusion_steps,
                dtype=torch.float32,
                device=self.device
            )
            return betas if self.noise_schedule == 'linear' else self._betas_from_linear_variance(variance=betas)
        elif self.noise_schedule == 'cosine':
            return self._betas_for_alpha_bar()
        elif self.noise_schedule == 'binomial':
            diffusion_steps_range = torch.arange(self.max_diffusion_steps, device=self.device)
            betas = [1.0 / (self.max_diffusion_steps - t + 1) for t in diffusion_steps_range]
            return torch.tensor(betas, device=self.device, dtype=torch.float32)
        else:
            raise NotImplementedError(f'Noise schedule {self.noise_schedule} not implemented')

    def diffusion_step_embedding(self, diffusion_steps: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(self.device)
        args = diffusion_steps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    @staticmethod
    def _extract_into_tensor(
            array: torch.Tensor,
            diffusion_steps: torch.Tensor,
            broadcast_shape: torch.Size
    ) -> torch.Tensor:
        result = array[diffusion_steps].float()
        while len(result.shape) < len(broadcast_shape):
            result = result[..., None]
        return result.expand(broadcast_shape)

    def add_gaussian_noise(self, x: torch.Tensor, diffusion_steps: torch.Tensor) -> torch.Tensor:
        gaussian_noise = torch.randn_like(x)
        return (
            self._extract_into_tensor(
                array=self.sqrt_alphas_cumprod,
                diffusion_steps=diffusion_steps,
                broadcast_shape=x.shape
            ) * x
            + self._extract_into_tensor(
                array=self.sqrt_one_minus_alphas_cumprod,
                diffusion_steps=diffusion_steps,
                broadcast_shape=x.shape
            ) * gaussian_noise
        )

    def q_posterior_mean_variance(
            self,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            diffusion_steps: torch.Tensor
    ) -> tuple:
        posterior_mean = (
            self._extract_into_tensor(
                array=self.posterior_mean_coef1,
                diffusion_steps=diffusion_steps,
                broadcast_shape=x_t.shape
            ) * x_start
            + self._extract_into_tensor(
                array=self.posterior_mean_coef2,
                diffusion_steps=diffusion_steps,
                broadcast_shape=x_t.shape
            ) * x_t
        )
        posterior_variance = self._extract_into_tensor(
            array=self.posterior_variance,
            diffusion_steps=diffusion_steps,
            broadcast_shape=x_t.shape
        )
        posterior_log_variance_clipped = self._extract_into_tensor(
            array=self.posterior_log_variance_clipped,
            diffusion_steps=diffusion_steps,
            broadcast_shape=x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            x: torch.Tensor,
            diffusion_steps: torch.Tensor,
            model_output: torch.Tensor
    ) -> tuple:
        model_variance = self._extract_into_tensor(
            array=self.posterior_variance,
            diffusion_steps=diffusion_steps,
            broadcast_shape=x.shape
        )
        model_log_variance = self._extract_into_tensor(
            array=self.posterior_log_variance_clipped,
            diffusion_steps=diffusion_steps,
            broadcast_shape=x.shape
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=model_output,
            x_t=x,
            diffusion_steps=diffusion_steps
        )

        return model_mean, model_variance, model_log_variance

    def forward_diffusion(self, **kwargs):
        raise NotImplementedError

    def reverse_diffusion(self, **kwargs):
        raise NotImplementedError
