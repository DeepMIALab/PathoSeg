#!/bin/bash -eu

# The eval.py scripts in the various pre-trained model folders differ slightly.
# * The model iteration is different
# * The range() statement that selects the checkpoint iteration sometimes includes +1 on end_iter, sometimes not
# * The image size differs
# * The batch size differs

case $1 in

    shell)
        model="trial_shell"
        start_iter=3
        end_iter=4
        size=1024
        batch_size=8
        ;;

    skull)
        model="trial_skull"
        start_iter=5
        end_iter=6
        size=1024
        batch_size=8
        ;;

    dog)
        model="trial_dog"
        start_iter=8
        end_iter=8
        size=256
        batch_size=8
        ;;

    art)
        model="good_art_1k_512"
        start_iter=5
        end_iter=5
        size=512
        batch_size=12
        ;;

    face)
        model="good_ffhq_full_512"
        start_iter=10
        end_iter=10
        size=512
        batch_size=16
        ;;

    *)
        echo "Unknown model '$1', valid options are: 'art', 'face', 'shell', 'skull', 'dog'."
        exit 1
        ;;
esac

shift 1

cd $model
python eval.py --start_iter=$start_iter --end_iter=$end_iter --im_size=$size --size=$size --batch=$batch_size $@
mv eval_${start_iter}0000/img/* /outputs/
