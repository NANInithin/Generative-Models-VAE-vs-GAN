# Generative-Models-VAE-vs-GAN
---

## 1. Introduction

This project explores two fundamental generative models: Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). Using the Fashion-MNIST dataset, we implemented both architectures to compare their image generation quality, latent space properties, and training stability. We further implemented a Wasserstein GAN (WGAN) to address stability issues found in the standard GAN, demonstrating improved training robustness and diversity in generated samples.

---

## 2. Methodology & Architectures

### 2.1 Variational Autoencoder (VAE)

**Architecture Overview:** A Convolutional VAE was implemented following the recommended approach in the assignment specifications.

**Encoder Component:**
- Input: 1×28×28 Fashion-MNIST images, normalized to [0, 1]
- Layer 1: Conv2d(1→32, kernel=3, stride=2, padding=1) → 32×14×14
- Layer 2: Conv2d(32→64, kernel=3, stride=2, padding=1) → 64×7×7
- Flatten and project to latent statistics via two fully connected layers

**Latent Space:**
- Dimensionality: 20 (reparameterization trick applied)
- Distribution: Standard normal N(0,I) prior

**Decoder Component:**
- Fully connected layer: Linear(20→64×7×7)
- Layer 1: ConvTranspose2d(64→32, kernel=3, stride=2, padding=1, output_padding=1) → 32×14×14
- Layer 2: ConvTranspose2d(32→1, kernel=3, stride=2, padding=1, output_padding=1) → 1×28×28
- Final activation: Sigmoid (outputs in [0, 1])

**Loss Function:**
The total loss combines reconstruction and regularization:

L_VAE = E[BCE(x, x̂)] + KL(N(μ, σ²) || N(0, I))

Where:
- BCE is computed per-pixel between input and reconstruction
- KL divergence penalizes deviation from the standard normal prior

**Training Specification:**
- Optimizer: Adam (lr=1e-3)
- Batch size: 128
- Epochs: 20
- Device: CUDA (RTX 4060)

### 2.2 Generative Adversarial Network (GAN) - Standard DCGAN

**Architecture Overview:** A DCGAN-style architecture with separate Generator and Discriminator networks.

**Generator:**
- Input: Noise vector z ∈ ℝ^100 (shape: [batch, 100, 1, 1])
- ConvTranspose2d(100→128, kernel=7, stride=1) → [batch, 128, 7, 7]
- BatchNorm2d + ReLU
- ConvTranspose2d(128→64, kernel=4, stride=2, padding=1, output_padding=1) → [batch, 64, 14, 14]
- BatchNorm2d + ReLU
- ConvTranspose2d(64→1, kernel=4, stride=2, padding=1, output_padding=1) → [batch, 1, 28, 28]
- Final activation: Tanh (outputs in [-1, 1])

**Discriminator:**
- Input: 1×28×28 images
- Conv2d(1→64, kernel=4, stride=2, padding=1) → 64×14×14
- LeakyReLU(0.2)
- Conv2d(64→128, kernel=4, stride=2, padding=1) → 128×7×7
- BatchNorm2d + LeakyReLU(0.2)
- Conv2d(128→1, kernel=7, stride=1) → [batch, 1, 1, 1]
- Final activation: Sigmoid (probability real vs. fake)

**Loss Function:**
Standard minimax loss:

L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
L_G = -E[log D(G(z))]

**Training Specification:**
- Optimizer: Adam (lr=2e-4, β₁=0.5)
- Batch size: 128
- Epochs: 30
- Alternating updates: 1 Discriminator update + 1 Generator update per iteration
- Data normalization: Normalize to [-1, 1] for Tanh compatibility

### 2.3 Stability Improvement: Wasserstein GAN (WGAN)

**Motivation:** Standard GANs suffer from mode collapse and training instability. WGAN addresses this by using the Wasserstein distance instead of the Jensen-Shannon divergence.

**Key Modifications:**

**Critic (Modified Discriminator):**
- Identical architecture to the standard discriminator **except**
- Removed final Sigmoid activation (now linear output)
- Output represents a score, not a probability

**Loss Function (Wasserstein Distance):**

L_D = -E[D(x)] + E[D(G(z))]  (maximize real score, minimize fake score)
L_G = -E[D(G(z))]             (maximize critic's score on fakes)

**Weight Clipping:**
After each critic gradient update, clip all parameters to [-0.01, 0.01] to enforce 1-Lipschitz continuity.

**Training Protocol:**
- Optimizer: RMSprop (lr=5e-5) instead of Adam
- Critic updates: 5 steps for every 1 generator step (N_CRITIC=5)
- Batch size: 64
- Epochs: 30

---

## 3. Experimental Results

### 3.1 VAE Training Performance

**Final Loss Metrics:**
- Epoch 20 Loss: 241.07 (averaged over training set)
- Loss composition: ~60% Reconstruction + ~40% KL divergence

**Observations:**
- Loss decreased steadily throughout training (no oscillations)
- Training was stable and predictable
- Generated samples showed increasing coherence from epoch 1 to epoch 20

### 3.2 Generated Samples & Interpolation

**Reconstruction Quality:**
The VAE successfully reconstructed input images with characteristic smooth appearance. While not pixel-perfect, the reconstructions preserved global shape and major features.

**Latent Space Interpolation:**
Linear interpolation between two latent vectors z₁ and z₂ using:

z(t) = (1-t)z₁ + tz₂,  t ∈ [0, 1]

Results showed smooth semantic transitions, e.g., morphing a shoe into a shirt through intermediate clothing types. This demonstrates that the VAE learned a continuous, well-structured latent manifold.

### 3.3 GAN vs. WGAN Comparison

| Aspect | Standard DCGAN | WGAN |
|--------|---|---|
| **Loss Stability** | Highly oscillatory; difficult to monitor convergence | Smooth, monotonic decrease in critic loss |
| **Image Sharpness** | Sharp edges, high-frequency details | Comparable sharpness, slightly smoother textures |
| **Diversity** | Prone to mode collapse after ~15 epochs | Consistently diverse across all epochs |
| **Training Robustness** | Sensitive to learning rate and batch size | Robust; less hyperparameter tuning required |
| **Convergence Speed** | Faster initial convergence | Slower but more reliable convergence |
| **Artifact Presence** | Some checkerboard artifacts visible | Fewer artifacts; cleaner output |

**Key Observation:**
The standard GAN showed signs of mode collapse by epoch 15-20, generating predominantly similar clothing items (e.g., mostly shoes or mostly t-shirts). WGAN maintained high diversity throughout, producing varied clothing types at every epoch.

---

## 4. Discussion

### 4.1 VAE Strengths & Limitations

**Strengths:**
- Provides a well-defined probabilistic model with interpretable latent space
- Enables smooth interpolation and controlled generation
- Training is stable and reproducible
- Suitable for downstream tasks like anomaly detection

**Limitations:**
- Generated images are characteristically blurry due to pixel-wise reconstruction loss (BCE/MSE)
- High variance in the KL term can lead to posterior collapse (latent variables ignored)
- Lower visual quality compared to GANs

### 4.2 GAN Observations

**Standard DCGAN:**
- Produces sharp, realistic images
- Training is unstable: discriminator often overpowers generator, causing gradient vanishing
- Mode collapse: Generator learns to map diverse noise inputs to similar outputs
- Difficult to monitor: No explicit loss metric indicates convergence

**WGAN Improvement:**
- The Wasserstein distance provides a meaningful loss metric even when distributions are disjoint
- Weight clipping enforces the Lipschitz constraint, preventing gradient explosion/vanishing
- Training the critic 5× per generator step allows the discriminator to reach "optimality" before updating the generator
- Result: More stable, robust training with fewer hyperparameter adjustments

### 4.3 Trade-offs & Model Selection

**When to use VAE:**
- Interpretable latent space required
- Generation with specific attributes needed (via latent interpolation)
- Stable training mandatory (no tolerance for instability)

**When to use GAN (or WGAN):**
- Maximum visual quality/sharpness required
- Computational resources available for longer training
- Mode collapse acceptable or addressed (via WGAN, spectral normalization, etc.)

---

## 5. Conclusion

This project successfully demonstrated the implementation and comparison of two major generative modeling paradigms:

1. **VAE** provided a principled probabilistic approach with excellent latent space structure but visually blurry outputs.

2. **Standard GAN** achieved sharper images but suffered from training instability and mode collapse.

3. **WGAN** successfully bridged these challenges, delivering both improved stability and maintained diversity—the optimal compromise for this dataset.

The Wasserstein distance metric and weight clipping proved crucial for robust training. Going forward, techniques like Spectral Normalization or Gradient Penalty could further enhance GAN training without the computational overhead of 5× critic updates.

---

## 6. References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
- Goodfellow, I., et al. (2014). Generative Adversarial Nets. *NeurIPS*.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *arXiv preprint arXiv:1701.07875*.
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *arXiv preprint arXiv:1511.06434*.

---

**Appendix: Code Snippets**

All code used in this project is available in the following files:
- `train_vae.py` - VAE training and latent space exploration
- `train_gan.py` - Standard DCGAN implementation
- `train_wgan.py` - Wasserstein GAN with weight clipping

Generated results are saved in:
- `results/vae_results/` - VAE outputs
- `results/gan_results/` - Standard GAN outputs
- `results/wgan_results/` - WGAN outputs
