

## Generative AI Research

One of our main goals for this project is to use generative AI to take our relatively small dataset and create synthetic data that is indistinguishable in terms of quality and accuracy from the original. We will then use this synthetic data to train our machine learning algorithm to accurately predict the next play in a football game.

After thorough research, we have concluded that in order to generate accurate synthetic data, it is imperative that it reflects the statistical patterns and relationships present in the original dataset. To start, we need to ensure that the dataset is processed and well-curated. The program must have all the accurate data it needs and be able to read it properly.

A few options we have for generating the synthetic data are:

Generative Adversarial Networks (GANs):

GANs are a class of machine learning frameworks effective at generating synthetic data. They work by creating two neural networks that contest with each other in the form of a zero-sum game, where one agent's gain is another agent's loss. The first neural network is called the generator model, which creates new samples, and the second model is called the discriminator model, which ensures that the generated samples are indistinguishable from real data. The biggest challenge for this model would be training the GAN to maintain the intricate relationships between variables in football (e.g., how closely related down and distance are, how the score affects how much you throw the ball, etc.).

Variational Autoencoders (VAEs):

VAEs are generative models specialized in creating variations of the input data they're trained on. They also perform tasks common to other encoders, such as denoising. VAEs specialize in probabilistic representation, where, unlike traditional autoencoders that map each input to a fixed point, VAEs map it to a distribution. We can then sample from this distribution to generate new data. Some potential downsides are that VAEs’ data, while generally correct, can be a bit "fuzzy," which is troublesome for us when trying to predict plays that rely on sharp data. For example, the odds of running the ball at 4th and 3 are very different from running the ball at 4th and 1, but the VAE may not pick up on that subtlety.