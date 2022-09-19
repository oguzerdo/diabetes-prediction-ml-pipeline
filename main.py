# End-to-End Diabetes Machine Learning Pipeline
# Author: Oguz Erdogan
# www.github.com/oguzerdo

import pandas as pd
from scripts.preprocess import *
from scripts.train import *

from contextlib import contextmanager
import time
import joblib

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    if (time.time() - t0) < 60:
        print("{} - done in {:.0f}s".format(title, time.time() - t0))
        print(" ")
    else:
        duration = time.time() - t0
        min = duration // 60
        second = int(duration - min * 60)
        print(f"{title} is finished in {min} min. {second} second")
        print(" ")


def main(debug=True, tuning=True):
    with timer("Pipeline"):
        print("Pipeline started")
        with timer("Reading Dataset"):
            print("Reading Dataset Started")
            df = pd.read_csv(DATA_PATH)

        with timer("Data Preprocessing"):
            print("Data Preprocessing Started")
            df = data_preprocessing(df)

        with timer("Training"):
            print("Training Started")
            final_model = train_model(debug, tuning)
            joblib.dump(final_model, 'outputs/final_model.pkl')

if __name__ == "__main__":
    namespace = get_namespace()
    with timer("Full model run"):
        main(debug=namespace.debug,
             tuning=namespace.tuning)
