from faker import Faker
import numpy as np
import pandas as pd


def main(num_rows=1000):
    fake = Faker()
    names = [fake.name() for _ in range(num_rows)]
    sensitive_feature1 = np.random.normal(loc=10, scale=100, size=num_rows)
    sensitive_feature2 = np.random.normal(loc=3, scale=10, size=num_rows)
    sensitive_feature3 = np.random.gamma(shape=2, scale=1, size=num_rows)
    sensitive_feature4 = np.random.exponential(scale=20, size=num_rows)
    df = pd.DataFrame({'name': names,
        'sensitive_feature1': sensitive_feature1,
        'sensitive_feature2': sensitive_feature2,
        'sensitive_feature3': sensitive_feature3,
        'sensitive_feature4': sensitive_feature4,
        })
    df.to_csv('sensitive_data.csv', index=None)


if __name__ == '__main__':
    main()
