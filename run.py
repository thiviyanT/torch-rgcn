from utils.misc import create_experiment


# Create sacred object for experiment tracking
ex = create_experiment(name='R-GCN Node Classification', database='node_class')

print(ex)