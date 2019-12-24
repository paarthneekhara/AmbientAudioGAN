# AmbientAudioGAN
Recently, Generative Adversarial Networks (GANs) have shown success in raw audio synthesis in an unsupervised setting. However, current techniques for audio synthesis using GANs require access to fully observed audio signals during training which may not be available for all tasks of practical interests. In this work, we introduce a framework called Audio-RecGAN, a generative adversarial training technique to explore different noisy measurement settings tailored for raw audio, where it is possible to recover the underlying data distribution from a training dataset containing only lossy audio signals. We further demonstrate the robustness of Audio-RecGAN in scenarios where the measurement model is not known exactly. Finally, we propose and investigate a novel optimization objective to project lossy audio signals from natural data space back to the latent space of the trained Audio-RecGAN generator, allowing us to recover clean audio signals.