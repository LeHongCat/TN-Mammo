import subprocess
import argparse

def train_all_folds(config_path, num_folds=5):
    for fold in range(num_folds):
        print(f"\nTraining fold {fold}")
        subprocess.run([
            "python", "train.py",
            "--config_path", config_path,
            "--fold", str(fold)
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='ThongNhat_config.yml')
    parser.add_argument('--num_folds', type=int, default=5)
    args = parser.parse_args()
    
    train_all_folds(args.config_path, args.num_folds) 