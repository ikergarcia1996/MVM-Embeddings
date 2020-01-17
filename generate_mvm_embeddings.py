import argparse
import os
from utils import (
    generate_dictionary_for_vecmap,
    print_dictionary_for_vecmap,
    printTrace,
    get_dimensions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--embeddings", nargs="+", required=True)
    parser.add_argument("-t", "--rotate_to", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-v", "--vocabulary", default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-k", "--num_nearest_neighbor", type=int, default=10)
    parser.add_argument("-r", "--retrofitting", default=None)
    parser.add_argument("-rn", "--retrofitting_n_iters", type=int, default=10)
    # parser.add_argument('-n', '--do_not_normalize_embs', default=False)
    parser.add_argument("-ir", "--do_not_retrofit_rotate_to", default=False)
    parser.add_argument("-nc", "--do_not_clean_files", default=False)
    parser.add_argument("-oov", "--generate_oov_words", action="store_false")

    args = parser.parse_args()

    is_rot_in_input = None

    for emb_i, emb in enumerate(args.embeddings):
        if emb == args.rotate_to:
            is_rot_in_input = emb_i

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    print(
        "tmp folder created, it will be deleted at the end of the execution (unless you have run the program with the -nc True option)"
    )

    if args.retrofitting is not None:
        printTrace("==> Retrofitting <==")
        for emb_i, emb in enumerate(args.embeddings):
            string = (
                str(emb_i + 1)
                + " of "
                + str(
                    len(args.embeddings)
                    if is_rot_in_input is not None or args.do_not_retrofit_rotate_to
                    else str(len(args.embeddings) + 1)
                )
            )
            print(string)
            excec_com = (
                "python3 Retrofitting/retrofit.py -i "
                + str(emb)
                + " -l "
                + str(args.retrofitting)
                + " -n "
                + str(args.retrofitting_n_iters)
                + " -o "
                + "tmp/"
                + str(emb_i)
                + ".retro -d "
                + str(get_dimensions(emb))
            )
            print(excec_com)
            os.system(excec_com)

        if is_rot_in_input is not None and not args.do_not_retrofit_rotate_to:
            string = (
                str(len(args.embeddings + 1)) + " of " + str(len(args.embeddings))
                if is_rot_in_input is not None or args.do_not_retrofit_rotate_to
                else str(len(args.embeddings) + 1)
            )
            print(string)
            excec_com = (
                "python3 Retrofitting/retrofit.py -i "
                + str(args.rotate_to)
                + " -l "
                + str(args.retrofitting)
                + " -n "
                + str(args.retrofitting_n_iters)
                + " -o "
                + "tmp/"
                + "out.retro  -d "
                + str(get_dimensions(emb))
            )
            print(excec_com)
            os.system(excec_com)

        print()

    printTrace("==> Generating dictionaries for the mapping <==")

    for emb_i, emb in enumerate(args.embeddings):
        string = str(emb_i + 1) + " of " + str(len(args.embeddings))
        print(string)
        print_dictionary_for_vecmap(
            "tmp/" + str(emb_i) + ".dict",
            generate_dictionary_for_vecmap(path1=emb, path2=args.rotate_to),
        )

    print()

    printTrace("==> Normalizing Embeddings <==")

    for emb_i, emb in enumerate(args.embeddings):
        string = (
            str(emb_i + 1)
            + " of "
            + str(
                len(args.embeddings)
                if is_rot_in_input is not None
                else str(len(args.embeddings) + 1)
            )
        )
        print(string)
        excec_com = (
            "python3 VecMap/normalize_embeddings.py unit center -i "
            + (emb if args.retrofitting is None else "tmp/" + str(emb_i) + ".retro")
            + " -o tmp/"
            + str(emb_i)
            + ".norm"
        )
        print(excec_com)
        os.system(excec_com)

    if is_rot_in_input is None:
        string = str(len(args.embeddings) + 1) + " of " + str(len(args.embeddings) + 1)
        print(string)
        excec_com = (
            "python3 VecMap/normalize_embeddings.py unit center -i "
            + (
                args.rotate_to
                if args.retrofitting is None or args.do_not_retrofit_rotate_to
                else "tmp/out.retro"
            )
            + " -o tmp/out.norm"
        )
        print(excec_com)
        os.system(excec_com)

    print()

    printTrace("==> Mapping Embeddings <==")

    for emb_i, emb in enumerate(args.embeddings):

        if is_rot_in_input is None or (
            is_rot_in_input is not None and is_rot_in_input != emb_i
        ):

            string = (
                str(emb_i + 1) + " of " + str(len(args.embeddings) - 1)
                if is_rot_in_input is not None
                else str(len(args.embeddings) + 1)
            )
            print(string)

            source_input = "tmp/" + str(emb_i) + ".norm"
            target_input = (
                "tmp/out.norm"
                if is_rot_in_input is None
                else "tmp/" + str(is_rot_in_input) + ".norm"
            )
            source_output = "tmp/" + str(emb_i) + ".vecmap"
            target_output = "tmp/out.vecmap"
            dictionary = "tmp/" + str(emb_i) + ".dict"

            excec_com = (
                "python3 VecMap/map_embeddings.py --orthogonal "
                + source_input
                + " "
                + target_input
                + " "
                + source_output
                + " "
                + target_output
                + " -d "
                + dictionary
            )
            print(excec_com)
            os.system(excec_com)

    print()

    printTrace("==> Generating Meta Embedding <==")

    embs = ""
    for emb_i, emb in enumerate(args.embeddings):
        if is_rot_in_input is None or (
            is_rot_in_input is not None and is_rot_in_input != emb_i
        ):
            embs = embs + "tmp/" + str(emb_i) + ".vecmap "

    if is_rot_in_input is not None:
        embs = embs + "tmp/out.vecmap "

    excec_com = (
        "python3 embeddings_mean.py -i "
        + embs
        + "-o "
        + args.output
        + " -b "
        + str(args.batch_size)
        + " -k "
        + str(args.num_nearest_neighbor)
    )

    if not args.generate_oov_words:
        excec_com = excec_com + " -oov"
    if args.vocabulary is not None:
        excec_com = excec_com + " -v " + args.vocabulary
    print(excec_com)
    os.system(excec_com)

    print()
    print("Done! Meta embedding generated in " + args.output)

    if not args.do_not_clean_files:
        print("Cleaning files...")
        try:
            os.system("rm -rf tmp")
        except:
            print("Could not delete the tmp folder, do it manually")


if __name__ == "__main__":
    main()
