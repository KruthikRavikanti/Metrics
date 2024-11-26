import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from collections import defaultdict
from pathlib import Path
import os

from train_model import make_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def infer_image_moded_xception(image_path, classes, model):
    # print(classes)
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis


    predictions = list(model.predict(img_array)[0])
    res = defaultdict(str)
    for i in range(len(predictions)):
        res[classes[i]] = str(round(predictions[i] * 100, 3)) + "%"
    return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))


@hydra.main(config_path='conf', config_name='classifier_inference', version_base='1.3.2')
def main(cfg: DictConfig):
    setattr(cfg, 'input_dir', Path(cfg.input_dir))
    setattr(cfg, 'output_dir', Path(cfg.output_dir))
    setattr(cfg, 'model_path', Path(cfg.model_path))
    classes = cfg.classes
    classes.sort()
    image_size = (224, 224)
    model = make_model(input_shape=image_size + (3,), num_classes=len(classes))
    model.load_weights(cfg.model_path)

    outputs = []
    for image_path in cfg.input_dir.glob('*png'):
        results = infer_image_moded_xception(image_path, classes, model)
        outputs.append({
            'image': image_path,
            'results': results,
            })

    with open(cfg.output_dir, 'w') as f:
        [json.dumps(l, f) for l in outputs]


if __name__ == "__main__":
    main()
