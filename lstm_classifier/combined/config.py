model_config = {
    'gpu': 0, # Original era 0
    'n_layers': 2,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 256,
    'input_dim': 2470, # Original era 2472
    'batch_size': 6000,  # carefully chosen original 200
    'n_epochs': 55000, # Original son 55000
    'learning_rate': 0.001,
    'bidirectional': True,
    'model_code': 'bi_lstm'
}
