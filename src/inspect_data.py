import os

data_dir = "../data/FER-2013"
sets = ["train", "test"]

for set_name in sets:
    path = os.path.join(data_dir, set_name)
    print(f"\n{set_name.upper()} SET:")

    for emotion in os.listdir(path):
        emotion_dir = os.path.join(path, emotion)
        if os.path.isdir(emotion_dir):
            num_images = len(os.listdir(emotion_dir))
            print(f"{emotion}: {num_images} images")
