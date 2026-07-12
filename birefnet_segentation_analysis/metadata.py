#
# metadata.py
#
#  This notebook contains metadata for the case studies used in the BirefNet segmentation EDA. 
#  Each case study includes a type (Success or Failure), a title, a description, and 
#  paths to the raw image and corresponding mask image.
#
#
CASE_STUDIES = [
    {
        "type": "Success",
        "title": "Case Study 1: General Example (Baseline Success)",
        "desc": "Demonstrates optimal baseline extraction under standard daylight conditions.",
        "raw": "loc_0121_im_003724.jpg",
        "mask": "hybrid_wolf_loc_0121_im_003724.png",
    },
    {
        "type": "Success",
        "title": "Case Study 2: Size Variation / Small Target",
        "desc": "Shows pipeline capability to extract small targets via soft-gated fallback cropping.",
        "raw": "5a0e378d-23d2-11e8-a6a3-ec086b02610b.jpg",
        "mask": "hybrid_wildlife_subject_5a0e378d-23d2-11e8-a6a3-ec086b02610b.png",
    },
    {
        "type": "Success",
        "title": "Case Study 3: Low Contrast / Nighttime",
        "desc": "Shows pipeline capability to extract targets under low-contrast nighttime conditions.",
        "raw": "loc_0219_im_000506.jpg",
        "mask": "hybrid_wolf_loc_0219_im_000506.png",
    },
    {
        "type": "Failure",
        "title": "Case Study 4: Extreme Visual Overlap & Camouflage",
        "desc": "Identifies mask fragmentation limits when thick foliage splits the target silhouette.",
        "raw": "59a17be3-23d2-11e8-a6a3-ec086b02610b.jpg",
        "mask": "hybrid_wildlife_subject_59a17be3-23d2-11e8-a6a3-ec086b02610b.png",
    },
    {
        "type": "Failure",
        "title": "Case Study 5: Multi-Animal Overlap",
        "desc": "Highlights boundary distortion and instance merging when multiple targets interact.",
        "raw": "loc_0188_im_002313.jpg",
        "mask": "hybrid_wolf_loc_0188_im_002313.png",
    },
]