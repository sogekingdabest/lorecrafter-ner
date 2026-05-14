import yaml


def load_config(config_path="configs/training.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cfg = config["training"]

    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["num_epochs"] = int(cfg["num_epochs"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_ratio"] = float(cfg["warmup_ratio"])
    cfg["max_length"] = int(cfg["max_length"])
    cfg["seed"] = int(cfg["seed"])
    cfg["train_split"] = float(cfg["train_split"])
    cfg["val_split"] = float(cfg["val_split"])
    cfg["test_split"] = float(cfg["test_split"])

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(yaml.dump(cfg, default_flow_style=False))
