import csv
import os
from pathlib import Path
import cv2
import streamlit as st
import demo.session_state as session_st
from image_captioning.utils.helpers import load_config


def get_list_of_images(match):
    return os.listdir(f"demo/data/val_samples/{match}/images/")


def get_info_from_eval_csv(match, img):
    with open(f"demo/data/val_samples/{match}/eval.csv") as f:
        content = csv.DictReader(f, delimiter=',')
        for row in content:
            if row["Relative document path in dataset"] == img.replace(".png", ""):
                predicted = row["Predicted labels"]
                ground_truth = row["Ground truth labels"]
                dist = row["Levenshtein distance"]
                return predicted, ground_truth, dist


def fill_content(predicted, ground_truth, dist):
    st.markdown("### Prediction:")
    st.write(predicted)
    st.markdown("### Ground truth:")
    st.write(ground_truth)
    st.markdown(f"### Distance: {dist}")


def buttons(prev, next_, images, session_state):
    if next_.button("Next"):
        if session_state.current_img + 1 < len(images):
            session_state.current_img += 1
        else:
            session_state.current_img = 0

    if prev.button("Previous"):
        if session_state.current_img - 1 < 0:
            session_state.current_img = len(images) - 1
        else:
            session_state.current_img -= 1


def main():
    st.sidebar.title('Validation set')
    page = st.sidebar.radio("", ["Statistics", "Full match", "Distance < 5", "Large distance"])
    session_state = session_st.get(current_img=0)

    encoder_cfg = load_config(Path(
        "workdir/checkpoints/effnetv2_l_300x400_transformer-encoder-decoder/latest/encoder_config.yml"))

    mapping = {"Statistics": "statistics",
               "Full match": "full_match",
               "Distance < 5": "good_match",
               "Large distance": "bad_match"}

    choice = mapping[page]
    if choice == "statistics":
        st.write("Will be filled later")
    else:
        images = get_list_of_images(choice)
        prev, idx, next_ = st.columns([1, 1, 1])
        buttons(prev, next_, images, session_state)
        idx.write(session_state.current_img + 1)

        img = cv2.imread(f"demo/data/val_samples/{choice}/images/{images[session_state.current_img]}")
        predicted, ground_truth, dist = get_info_from_eval_csv(choice, images[session_state.current_img])
        if st.checkbox("See resized image"):
            dsize = tuple(reversed(encoder_cfg.size))  # width, height
            img = cv2.resize(img, dsize=dsize)

        st.image(img)
        fill_content(predicted, ground_truth, dist)


if __name__ == "__main__":
    main()
