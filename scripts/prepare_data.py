import os, shutil, glob

ROOT = os.path.dirname(os.path.dirname(__file__))

foreback_root = os.path.join(ROOT, "data", "foreback_raw")
actions_root = os.path.join(ROOT, "data", "actions_raw",
                            "Tennis Player Actions Dataset for Human Pose Estimation")

ssl_root = os.path.join(ROOT, "data_ssl")
down_root = os.path.join(ROOT, "data_downstream")

os.makedirs(os.path.join(ssl_root, "images"), exist_ok=True)
os.makedirs(os.path.join(down_root, "forehand"), exist_ok=True)
os.makedirs(os.path.join(down_root, "backhand"), exist_ok=True)

# 1) SSL pool: ALL FH/BH images from foreback_raw (unlabeled)
for img_path in glob.glob(os.path.join(foreback_root, "**", "*.jpg"), recursive=True):
    dest = os.path.join(ssl_root, "images", os.path.basename(img_path))
    if not os.path.exists(dest):
        shutil.copy(img_path, dest)

# 2) Downstream: FH/BH from Tennis Player Actions dataset
for cls in ["forehand", "backhand"]:
    src_dir = os.path.join(actions_root, "images", cls)
    dst_dir = os.path.join(down_root, cls)
    os.makedirs(dst_dir, exist_ok=True)
    for img_path in glob.glob(os.path.join(src_dir, "*.jpeg")):
        dest = os.path.join(dst_dir, os.path.basename(img_path))
        if not os.path.exists(dest):
            shutil.copy(img_path, dest)

print("Done! data_ssl/ and data_downstream/ created.")
