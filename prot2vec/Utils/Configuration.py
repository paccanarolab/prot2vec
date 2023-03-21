import configparser
import os

CONFIG = configparser.ConfigParser()
RUN_CONFIG = configparser.ConfigParser()
PFP_LR_CONFIG = configparser.ConfigParser()


def get_list(config, section, key, sep="\n"):
    def get_list_(section, key, fallback=None):
        value = config.get(section, key, fallback=fallback)
        list_value = None
        if isinstance(value, str):
            list_value = value.split(sep)
        else:
            list_value = value
        return list_value
    return get_list_


def get_config_item(config, dtype, section, key, default="error"):
    getfunc = None
    if dtype == "int":
        getfunc = config.getint
    elif dtype == "float":
        getfunc = config.getfloat
    elif dtype == "bool":
        getfunc = config.getboolean
    elif dtype == "list":
        getfunc = get_list(config, section, key)
    else: # string or path
        # TODO: check that paths exists and maybe add an option to create
        # the path if it doesn't exist
        getfunc = config.get

    if default == "error":
        val = getfunc(section, key)
    else:
        val = getfunc(section, key, fallback=default)
    if val is None:
        raise ValueError(
            f"configuration file muse set a value of type {dtype} for key \"{key}\" in section \"{section}\" ")

    return val

def load_pfp_lr(config_file):
    PFP_LR_CONFIG.read(config_file)
    run_config_file = os.path.expanduser(PFP_LR_CONFIG.get('configuration', 'run_config_file'))
    config = load_run(run_config_file)
    defaults = {
        "int":[
            ("training", "min_protein_count_per_term", 5),
        ]
    }
    mandatory_no_default = {
        "path":[
            ("configuration", "run_config_file"),
            ("training", "function_assignment_bp"),
            ("training", "function_assignment_mf"),
            ("training", "function_assignment_cc"),
            ("predictions", "dir_lr_out"),
            ("predictions", "predictions_out_file"),
        ],
    }

    for dtype, configs in mandatory_no_default.items():
        for section, key in configs:
            if section not in config:
                config[section] = {}
            config[section][key] = get_config_item(PFP_LR_CONFIG, dtype, section, key)

    for dtype, configs in defaults.items():
        for section, key, fallback in configs:
            if section not in config:
                config[section] = {}
            config[section][key] = get_config_item(PFP_LR_CONFIG, dtype, section, key, default=fallback)

    return config

def load_run(run_config):
    RUN_CONFIG.read(run_config)
    config_file = os.path.expanduser(RUN_CONFIG.get('configuration', 'config_file'))
    CONFIG.read(config_file)
    config = {}
    defaults = {
        "int":[
            ("dataset", "num_pca", -1),
            ("model", "batch_size_train", 256),
            ("model", "batch_size_val", 256),
            ("model", "batch_size_test", 256),
            ("model", "dim_first_hidden_layer", 1024),
            ("training", "num_epochs", 200),
        ],
        "float":[
            ("optimizer", "learning_rate", 0.0006),
            ("loss", "main_task_factor", 1.0),
        ],
        "str":[
            ("model", "alias", "infer"),
            ("dataset", "interpro_filename", "interpro.tab"),
            ("dataset", "semantic_similarity_filename", "semantic-similarity.tab"),
        ],
        "list":[
            ("dataset", "string_columns", [
                "neighborhood",
                "fusion",
                "cooccurence",
                "coexpression",
                "experiments",
                "database",
                "textmining",
            ]),
        ],
        "bool":[
            ("dataset", "negative_sampling", True),
            ("dataset", "combine_string", True),
            ("dataset", "include_homology", True),
            ("dataset", "include_biogrid", True),
            ("dataset", "interpro_pca", False),
            ("training", "output_debug", False),
        ]
    }
    mandatory_no_default = {
        "path":[
            ("configuration", "config_file"),
            ("model", "dir_model_output"),
            ("training", "dir_training_out"),
            ("dataset", "dir_indices"),
            ("dataset", "dir_train"),
            ("dataset", "dir_val"),
            ("dataset", "dir_test"),
            ("representations", "dir_representations_out"),
        ],
    }


    for dtype, configs in mandatory_no_default.items():
        for section, key in configs:
            if section not in config:
                config[section] = {}
            config[section][key] = get_config_item(RUN_CONFIG, dtype, section, key)

    for dtype, configs in defaults.items():
        for section, key, fallback in configs:
            if section not in config:
                config[section] = {}
            config[section][key] = get_config_item(RUN_CONFIG, dtype, section, key, default=fallback)

    return config

def save_run(run_config):
    RUN_CONFIG.write(run_config)


def load_configuration(config_file):
    CONFIG.read(config_file)


def save(config_file):
    CONFIG.write(config_file)


