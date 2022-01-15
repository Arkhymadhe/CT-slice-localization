import matplotlib.pyplot as plt
from jupyterthemes import jtplot

def visualize_predictions(y, y_hat, h, w, style):
    """ Visualize model predictions. """
    jtplot.style(style)
    
    plt.figure(figsize = (h, w))
    sns.scatterplot(y, y, color = 'blue')
    sns.scatterplot(y, y_hat, color = 'red')
    plt.show(); plt.close('all')
    
    return None