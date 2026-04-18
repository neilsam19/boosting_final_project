import numpy as np
from sklearn.datasets import make_moons, make_classification, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name="moons", n_samples=300, random_state=42):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    
    elif name == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            flip_y=0.0,
            random_state=random_state
        )
    
    elif name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        # now converting it to binary
        y = (y == 0).astype(int)
    
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        # now converting it to binary
        y = (y == 0).astype(int)

    else:
        raise ValueError("Unknown dataset")

    # Convert labels to {-1, +1}
    y = np.where(y == 0, -1, 1)

    return X, y


def add_label_noise(y, noise_level, random_state=42):
    np.random.seed(random_state)
    y_noisy = y.copy()
    n = len(y)

    n_flip = int(noise_level * n)
    flip_indices = np.random.choice(n, n_flip, replace=False)

    y_noisy[flip_indices] *= -1
    return y_noisy


def prepare_data(
    dataset="moons",
    noise_level=0.0,
    test_size=0.3,
    scale=True,
    random_state=42
):
    # Load dataset
    X, y = load_dataset(dataset, random_state=random_state)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Add noise ONLY to training labels
    y_train_noisy = add_label_noise(y_train, noise_level, random_state)

    # scaling (important for real datasets)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train_noisy, y_test