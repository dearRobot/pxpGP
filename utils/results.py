from matplotlib import pyplot as plt
import os


def plot_1d_result(train_x, train_y, test_x, mean, lower, upper, rank: int = 0):

    plt.figure(figsize=(12, 6))
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Train Data')
    plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean Prediction')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), 
                        alpha=0.5, color='blue', label='Confidence Interval')
    plt.title('Gaussian Process Regression Rank {}'.format(rank))
    plt.legend()
    plt.show()


def plot_2d_result(train_x, train_y, test_x, mean, lower, upper, rank: int = 0):

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(test_x[:, 0].cpu().numpy(), test_x[:, 1].cpu().numpy(), mean.cpu().numpy(), cmap='viridis', 
                     edgecolor='k')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Mean Prediction')

    ax.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), train_y.cpu().numpy(), color='k', marker='*', label='Train Data')
    ax.title('Gaussian Process Regression Rank {}'.format(rank))
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_result(train_x, train_y, test_x, mean, lower, upper, input_dim: int = 1, rank: int = 0):
    """
    Plot the results of Gaussian Process Regression.
    Args:
        train_x: Training input data.
        train_y: Training output data.
        test_x: Testing input data.
        mean: Predicted mean from the GP model.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        rank: Process rank (default: 0).
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("Rank must be a non-negative integer.")

    if input_dim==1:
        plot_1d_result(train_x, train_y, test_x, mean, lower, upper, rank)
        
    
    elif input_dim == 2:
        plot_2d_result(train_x, train_y, test_x, mean, lower, upper, rank)
        


    # plt.figure(figsize=(12, 6))
    # plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Train Data')
    # plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean Prediction')
    # plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), 
    #                     alpha=0.5, color='blue', label='Confidence Interval')
    # plt.title('Gaussian Process Regression Rank {}'.format(rank))
    # plt.legend()
    # plt.show()




def plot_pxpGP_result(train_x, train_y, inducing_pt, test_x, mean, lower, upper, rank: int = 0):
    """
    Plot the results of PXP-GP Regression.
    Args:
        train_x: Training input data.
        train_y: Training output data.
        inducing_pt: Inducing points.
        test_x: Testing input data.
        mean: Predicted mean from the PXP-GP model.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        rank: Process rank (default: 0).
    """
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("Rank must be a non-negative integer.")

    plt.figure(figsize=(12, 6))
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Train Data')
    
    # these are just x locations 
    plt.scatter(inducing_pt.detach().cpu().numpy(), [0] * len(inducing_pt), color='red', label='Inducing Points')  # Scatter plot for inducing points
    

    plt.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean Prediction')
    plt.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), 
                        alpha=0.5, color='blue', label='Confidence Interval')
    plt.title('PXP-GP Regression Rank {}'.format(rank))
    plt.legend()
    plt.show()