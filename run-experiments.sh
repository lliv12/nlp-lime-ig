# ---------------------------  REVIEWS  --------------------------- #
# Categorical
# !python train.py -d reviews -n reviews_cat_tk500_sq1500 -tk reviews_tokenizer_500 -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_cat_tk1000_sq1500 -tk reviews_tokenizer -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_cat_tk5000_sq1500 -tk reviews_tokenizer_5000 -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4

# !python train.py -d reviews -n reviews_cat_tk500_sq1000 -tk reviews_tokenizer_500 -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_cat_tk1000_sq1000 -tk reviews_tokenizer -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_cat_tk5000_sq1000 -tk reviews_tokenizer_5000 -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4

python train.py -d reviews -n reviews_cat_tk500_sq500 -tk reviews_tokenizer_500 -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d reviews -n reviews_cat_tk1000_sq500 -tk reviews_tokenizer -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d reviews -n reviews_cat_tk5000_sq500 -tk reviews_tokenizer_5000 -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2

# Binary
# !python train.py -d reviews -n reviews_bin_tk500_sq1500 -tk reviews_tokenizer_500 -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_bin_tk1000_sq1500 -tk reviews_tokenizer -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_bin_tk5000_sq1500 -tk reviews_tokenizer_5000 -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4

# !python train.py -d reviews -n reviews_bin_tk500_sq1000 -tk reviews_tokenizer_500 -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_bin_tk1000_sq1000 -tk reviews_tokenizer -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d reviews -n reviews_bin_tk5000_sq1000 -tk reviews_tokenizer_5000 -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4

python train.py -d reviews -n reviews_bin_tk500_sq500 -tk reviews_tokenizer_500 -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d reviews -n reviews_bin_tk1000_sq500 -tk reviews_tokenizer -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d reviews -n reviews_bin_tk5000_sq500 -tk reviews_tokenizer_5000 -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2

# ---------------------------  ESSAYS  --------------------------- #
# Categorical
# !python train.py -d essays -n essays_cat_tk500_sq1500 -tk essays_tokenizer_500 -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_cat_tk1000_sq1500 -tk essays_tokenizer -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_cat_tk5000_sq1500 -tk essays_tokenizer_5000 -b 32 -s categorical -sq 1500 -log -e 20 -em 32 -ah 2 -l 4

# !python train.py -d essays -n essays_cat_tk500_sq1000 -tk essays_tokenizer_500 -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_cat_tk1000_sq1000 -tk essays_tokenizer -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_cat_tk5000_sq1000 -tk essays_tokenizer_5000 -b 32 -s categorical -sq 1000 -log -e 20 -em 32 -ah 2 -l 4

python train.py -d essays -n essays_cat_tk500_sq500 -tk essays_tokenizer_500 -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d essays -n essays_cat_tk1000_sq500 -tk essays_tokenizer -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d essays -n essays_cat_tk5000_sq500 -tk essays_tokenizer_5000 -b 32 -s categorical -sq 500 -log -e 10 -em 32 -ah 2 -l 2

# Binary
# !python train.py -d essays -n essays_bin_tk500_sq1500 -tk essays_tokenizer_500 -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_bin_tk1000_sq1500 -tk essays_tokenizer -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_bin_tk5000_sq1500 -tk essays_tokenizer_5000 -b 32 -s binary -sq 1500 -log -e 20 -em 32 -ah 2 -l 4

# !python train.py -d essays -n essays_bin_tk500_sq1000 -tk essays_tokenizer_500 -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_bin_tk1000_sq1000 -tk essays_tokenizer -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_bin_tk5000_sq1000 -tk essays_tokenizer_5000 -b 32 -s binary -sq 1000 -log -e 20 -em 32 -ah 2 -l 4

python train.py -d essays -n essays_bin_tk500_sq500 -tk essays_tokenizer_500 -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d essays -n essays_bin_tk1000_sq500 -tk essays_tokenizer -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2
python train.py -d essays -n essays_bin_tk5000_sq500 -tk essays_tokenizer_5000 -b 32 -s binary -sq 500 -log -e 10 -em 32 -ah 2 -l 2

# Standardized
# python train.py -d essays -n essays_std_tk1000_sq500 -tk essays_tokenizer_1000 -b 32 -s standardized -sq 500 -log -e 10 -em 32 -ah 2 -l 2

# !python train.py -d essays -n essays_std_tk1000_sq1000 -tk essays_tokenizer_1000 -b 32 -s standardized -sq 1000 -log -e 20 -em 32 -ah 2 -l 4
# !python train.py -d essays -n essays_std_tk1000_sq1500 -tk essays_tokenizer_1000 -b 32 -s standardized -sq 1500 -log -e 20 -em 32 -ah 2 -l 4