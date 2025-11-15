import torch



def test_univariate_scalar_input(loc=0.5, variance=0.1):

    mu = torch.tensor(loc)

    sigma = torch.tensor(variance)

    distribution = torch.distributions.MultivariateNormal(mu, torch.eye(1) * sigma)

    sample = distribution.sample()

    print(sample)



def test_univariate_scalar_input_with_args_validation(loc=0.5, variance=0.1):

    mu = torch.tensor(loc)

    sigma = torch.tensor(variance)

    distribution = torch.distributions.MultivariateNormal(mu, torch.eye(1) * sigma, validate_args=True)

    sample = distribution.sample()

    print(sample)



def test_univariate_input(loc=([0.5]), variance=0.1):

    mu = torch.tensor(loc)

    sigma = torch.tensor(variance)

    distribution = torch.distributions.MultivariateNormal(mu, torch.eye(1) * sigma)

    sample = distribution.sample()

    print(sample)



def test_univariate_input_with_args_validation(loc=([0.5]), variance=0.1):

    mu = torch.tensor(loc)

    sigma = torch.tensor(variance)

    distribution = torch.distributions.MultivariateNormal(mu, torch.eye(1) * sigma, validate_args=True)

    sample = distribution.sample()

    print(sample)





if __name__ == "__main__":

    test_univariate_scalar_input(loc=0.5, variance=0.1)  # Crashes with Floating point exception (SIGFPE)

    #test_univariate_scalar_input_with_args_validation(loc=0.5, variance=0.1)  #Crashes with Floating point exception (SIGFPE)

    #test_univariate_input(loc=([0.5]), variance=0.1)  # Runs without errors. Haven't verified if samples are from the correct normal distribution

    #test_univariate_input_with_args_validation(loc=([0.5]), variance=0.1)  # Runs without errors. Haven't verified if samples are from the correct normal distribution