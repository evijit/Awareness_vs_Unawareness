import hydra
from omegaconf import DictConfig
from utils import load_dataset, create_model
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import pandas as pd
import os


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = load_dataset(cfg.dataset_name)
    train_dataset, test_dataset = dataset.split(2, seed=123)
    print(f"Loaded dataset: {cfg.dataset_name}")
    print(f"Using model: {cfg.model_name}")

    tf.reset_default_graph()
    with tf.Session() as sess:
        print(f"Initializing model...")
        model = create_model(
            cfg.model_name,
            cfg.dataset_name,
            dataset,
            sess=sess,
            **cfg.model_params if cfg.model_params is not None else {}
        )
        print(f"Training model...")
        model.fit(train_dataset)
        print("Done!")

        print(f"Evaluating model...")
        train_preds = model.predict(train_dataset)
        test_preds = model.predict(test_dataset)
        tdf = test_preds.convert_to_dataframe()[0]
        print(tdf)
        print('../../'+cfg.model_name+'_'+cfg.dataset_name+'_preds.csv')
        tdf.to_csv('../../'+cfg.model_name+'_'+cfg.dataset_name+'_preds.csv')
        
        print("Done!")

        # TODO: compute metrics


if __name__ == '__main__':
    main()
