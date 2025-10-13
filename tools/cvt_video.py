import cv2 as cv
import os
import os.path as osp
from tqdm import tqdm
import imageio


def convert(
    src, dst_dir, h_range=None, w_range=None, target_height=480, skip=4, max_frame_N=100
):
    os.makedirs(dst_dir, exist_ok=True)
    save_dir = osp.join(dst_dir, "images")
    os.makedirs(save_dir, exist_ok=True)

    # use opencv to read video frames
    cap = cv.VideoCapture(src)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"frame_count: {frame_count}")
    frame_list = []
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        if h_range is not None:
            frame = frame[h_range[0] : h_range[1], :]
        if w_range is not None:
            frame = frame[:, w_range[0] : w_range[1]]
        # resize frame to have height of target_height
        h, w = frame.shape[:2]
        target_width = int(w / h * target_height)
        frame = cv.resize(frame, (target_width, target_height))
        # cv.imwrite("./debug.png", frame)
        frame_list.append(frame)
    frame_list = frame_list[::skip]
    frame_list = frame_list[:max_frame_N]
    for i, frame in enumerate(frame_list):
        cv.imwrite(osp.join(save_dir, f"{i:05d}.jpg"), frame)

    return


if __name__ == "__main__":
    # convert("../data/film/inception_clip1.mp4", "../data/film/inceoption_clip1", )
    # convert("../data/film/tokyo-walk-clip.mp4", "../data/film/tokyo-walk-clip", )
    # convert("../data/film/tokyo-in-the-snow-clip.mp4", "../data/film/tokyo-in-the-snow-clip", )
    # convert("../data/film/suv-in-the-dust.mp4", "../data/film/suv-in-the-dust", )
    # convert("../data/film/chinese-new-year-dragon.mp4", "../data/film/chinese-new-year-dragon", )

    # convert("../data/wild/chinese-new-year-dragon.mp4", "../data/wild/chinese-new-year-dragon")
    
    # convert(
    #     "../data/wild/tokyo-in-the-snow-clip.mp4",
    #     "../data/wild/tokyo-in-the-snow-clip",
    #     # skip=1,
    #     # max_frame_N=400,
    # )
    
    convert(
        "../../spatracker/assets/butterfly.mp4",
        "../../spatracker/assets/butterfly",
        # "../data/wild/tokyo-in-the-snow-clip",
        skip=1,
        max_frame_N=400,
    )
