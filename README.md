# fast MRI project

### models/
- ft_cnn_models.py: standard CNN for image reconstruction

- ft_attcnn_models.py: standard CNN with attention machanims (conditioned on mask or not) to select input information and addon to the estimated information

- ft_vaenn_models.py: learn posterior/prior networks and encode learned latent variables in the results of a reconstruction network

- ft_caenn_models.py: add cvae to learn the distributions of invisiable information

## Notes


## TODO
test_moving_ratio:

test_kspace_recom: