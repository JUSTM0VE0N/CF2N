
config = {
    # Training Configuration
    'validation': True,

    # Satellite configuration
    'satellite': 'PRISMA',
    'ratio': 6,
    'nbits': 16,

    # Training settings
    'save_weights': True,
    'save_weights_path': 'weights',
    'save_training_stats': False,

    # Training hyperparameters
    'learning_rate': 0.00001,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epochs': 200,
    'batch_size': 4,
    'semi_width': 18,

    'alpha_1': 0.5,
    'alpha_2': 0.25,

    'first_iter': 20,
    'epoch_nm': 15,
    'sat_val': 80,

    'net_scope': 6,
    'ms_scope': 1,

}