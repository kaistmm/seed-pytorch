import os
import argparse

# Set vox1-O path and test
# voxceleb1_path = "./voxceleb1/"
# voxceleb1_test = "./manifests/vox1-O.txt"

# voxconverse_path = "./voxconverse_test_SV/"
# voxconverse_test = "./voxconverse_test_SV/trials_wo_overlap.txt"
# target_test_fn = "./manifests/vcmix_test.txt"

def get_target_test_list(test_list, target="1"):
    with open(test_list, "r") as file:
        lines = file.readlines()
        
    target_audio = [line.strip().split() for line in lines if line.strip().split()[0] == target]

    print(f"Num of target test in {os.path.basename(test_list)}: {len(target_audio)}")
    return target_audio

def write_test(trg_file, audio_list, base_path, use_absolute_path=True):
    missing_count = 0
    for audio in audio_list:
        answer, audio_path1, audio_path2 = audio
        
        if use_absolute_path:
            audio_path1_ = os.path.abspath(os.path.join(base_path, audio_path1))
            audio_path2_ = os.path.abspath(os.path.join(base_path, audio_path2))
        else:
            audio_path1_ = os.path.join(base_path, audio_path1)
            audio_path2_ = os.path.join(base_path, audio_path2)

        
        missing_count += not os.path.isfile(audio_path1_)
        missing_count += not os.path.isfile(audio_path2_)

        trg_file.write(f"{answer} {audio_path1} {audio_path2}\n")
    return missing_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mixed test list from VoxCeleb1 and VoxConverse.")
    parser.add_argument('--voxceleb1_path',                 type=str, default="./voxceleb1/",           help='Path to VoxCeleb1 dataset.')
    parser.add_argument('--voxceleb1_test',                 type=str, default="./manifests/vox1-O.txt", help='Path to VoxCeleb1 test list file.')
    parser.add_argument('--voxconverse_path',               type=str, default="./voxconverse_test_SV/", help='Path to VoxConverse test dataset.')
    parser.add_argument('--voxconverse_test',               type=str, default="./voxconverse_test_SV/trials_wo_overlap.txt", help='Path to VoxConverse test list file.')
    parser.add_argument('--output_filename',                type=str, default="./manifests/vcmix_test.txt", help='Output path for the mixed test list file.')
    parser.add_argument('--download_voxconverse_test_SV',   default=False, action="store_true", help='Download VoxConverse test SV dataset.')

    args = parser.parse_args()

    voxceleb1_path   = args.voxceleb1_path
    voxceleb1_test   = args.voxceleb1_test
    voxconverse_path = args.voxconverse_path
    voxconverse_test = args.voxconverse_test
    output_filename  = args.output_filename

    if args.download_voxconverse_test_SV:
        os.system("wget --no-check-certificate https://mm.kaist.ac.kr/projects/seed-pytorch/voxconverse_test_SV.zip")
        os.system("unzip  ./voxconverse_test_SV.zip")
        os.system("rm -rf ./voxconverse_test_SV.zip")
        print("Downloaded VoxConverse test SV dataset into ./voxconverse_test_SV/ directory.")
        

    voxconverse_target_test = get_target_test_list(voxconverse_test, "0")
    voxceleb_target_test = get_target_test_list(voxceleb1_test, "1")

    missing_files = 0
    with open(output_filename, "w") as trg_file:
        missing_files += write_test(trg_file, voxconverse_target_test, voxconverse_path, use_absolute_path=False)
        missing_files += write_test(trg_file, voxceleb_target_test, voxceleb1_path, use_absolute_path=False)

    print(f"Number of missing files: {missing_files}")

    print("If you can see the output log as follows: \n \
          Num of target test in trials_wo_overlap.txt: 140734 \n \
          Num of target test in cleaned_test_list.txt: 18802 \n \
          Number of missing files: 0 \n \
          Then, you are successful to make the `vcmix_test.txt` file.")
