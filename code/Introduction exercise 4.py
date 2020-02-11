import os
import matplotlib.pyplot as plt
import random

base_path = 'C:/Users/max/stack/TUE/Sync_laptop/Imaging project/.data'
train_path_true = os.path.join(base_path, 'train+val', 'train', '1')
train_path_false = os.path.join(base_path, 'train+val', 'train', '0')
list_train_true = os.listdir(train_path_true)
list_train_false = os.listdir(train_path_false)

for i in range(1, 6, 2):

    plt.subplot(3, 2, i)
    false_path = random.choice(list_train_false)
    false = plt.imread(os.path.join(train_path_false, false_path))
    plt.imshow(false)
    plt.axis('off')
    if i == 1:
        plt.title('False')

    plt.subplot(3, 2, i + 1)
    true_path = random.choice(list_train_true)
    true = plt.imread(os.path.join(train_path_true, true_path))
    plt.imshow(true)
    plt.axis('off')
    if i+1 == 2:
        plt.title('True')

plt.show()
