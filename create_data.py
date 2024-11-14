import numpy as np
import cv2
import glob
from empatches import EMPatches
from modules.video_io import VideoReader
from modules.optical_flow import RAFT_Wrapper
from modules.utils import createDir, savePickle
import os
from tqdm import tqdm

def createData(config, isTrain=False):
    chunkSize = config["DATA_CHUNK_SIZE"] * 1024 * 1024 # Converting MB into Bytes
    maxFrames = config["MAX_FRAMES_PER_VIDEO"]
    patchSize = config["PATCH_SIZE"]
    stride = config["PATCH_STRIDE"]
    neighborhood = config["NEIGHBORHOOD"]
    output_path = config["DATA_STORE_PATH"]
    input_path = config["RAW_VIDEO_PATH"]

    createDir(output_path)

    if isTrain:
        input_path = os.path.join(input_path, "train")
        output_path = os.path.join(output_path, "train")
    else:
        input_path = os.path.join(input_path, "test")
        output_path = os.path.join(output_path, "test")

    createDir(output_path)

    indices_path = os.path.join(output_path, "indices.pkl")

    src_output_path = os.path.join(output_path, "src")
    createDir(src_output_path)
    dst_output_path = os.path.join(output_path, "dst")
    createDir(dst_output_path)
    mask_output_path = os.path.join(output_path, "mask")
    createDir(mask_output_path)

    patch_size_bytes = patchSize * patchSize * 4

    current_chunk_src = None
    current_chunk_dst = None
    current_chunk_mask = None
    index = 0

    size_used = 0

    emp = EMPatches()
    pwcnet = RAFT_Wrapper()
    indices = {}

    prev_range_end = -1
    cur_range_end = -1

    try:
        for file in glob.glob(os.path.join(input_path, "*")):
            reader = VideoReader(file, gray=False)
            frames = reader.getAllFrames(maxFrames=maxFrames)
            reader.close()

            print("Processing {}:".format(file))
            prog_bar = tqdm(range(neighborhood // 2, len(frames) - neighborhood // 2 - 1))
            for i in prog_bar:
                if(i % 20 != 0):
                    continue
                src = frames[i]
                src_g = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
                # src_g = src_g.astype('float') / 255.
                #########################
                # src_patches, _ = emp.extract_patches(src_g, patchsize=patchSize, stride=patchSize)
                # src_patches = np.array(src_patches)
                # src_patches = np.reshape(src_patches, (src_patches.shape[0], -1, 1))
                #########################
                for g in range(i-neighborhood//2, i+neighborhood//2+1):
                    dst = frames[g]
                    dst_warped = pwcnet.warpIntoAnother(dst, src)
                    dst_warped = cv2.cvtColor(dst_warped, cv2.COLOR_RGB2GRAY)
                    # dst_warped = dst_warped.astype('float') / 255.

                    ########################################
                    ## Mask Code ##

                    mask = pwcnet.generate_occlusions_mask(src=dst, dst=src)
                    # src_gm = src_g * mask
                    # dst_warped = dst_warped * mask
                    # dst_warped[mask == 0] = src_g[mask == 0]

                    src_patches, _ = emp.extract_patches(src_g, patchsize=patchSize, stride=patchSize)
                    src_patches = np.array(src_patches)
                    src_patches = np.reshape(src_patches, (src_patches.shape[0], -1, 1))

                    mask_patches, _ = emp.extract_patches(mask, patchsize=patchSize, stride=patchSize)
                    mask_patches = np.array(mask_patches)
                    mask_patches = np.reshape(mask_patches, (mask_patches.shape[0], -1, 1))

                    ########################################

                    dst_patches, _ = emp.extract_patches(dst_warped, patchsize=patchSize, stride=patchSize)
                    dst_patches = np.array(dst_patches)
                    dst_patches = np.reshape(dst_patches, (dst_patches.shape[0], -1, 1))

                    if current_chunk_src is None:
                        current_chunk_src = np.copy(src_patches)
                        current_chunk_dst = np.copy(dst_patches)
                        current_chunk_mask = np.copy(mask_patches)
                    else:
                        current_chunk_src = np.append(current_chunk_src, src_patches, axis=0)
                        current_chunk_dst = np.append(current_chunk_dst, dst_patches, axis=0)
                        current_chunk_mask = np.append(current_chunk_mask, dst_patches, axis=0)

                    cur_range_end += dst_patches.shape[0]

                    cur_chunk_size = current_chunk_src.size * current_chunk_src.itemsize
                    if(cur_chunk_size >= chunkSize):
                        np.save(os.path.join(src_output_path, "{}.npy".format(index)), current_chunk_src)
                        np.save(os.path.join(dst_output_path, "{}.npy".format(index)), current_chunk_dst)
                        np.save(os.path.join(mask_output_path, "{}.npy".format(index)), current_chunk_mask)

                        for i in range(prev_range_end + 1, cur_range_end + 1):
                            indices[i] = (index, prev_range_end+1)
                        
                        savePickle(indices_path, indices)
                        prev_range_end = cur_range_end
                        current_chunk_src = None
                        current_chunk_dst = None
                        current_chunk_mask = None
                        index += 1
                        size_used += cur_chunk_size
                        prog_bar.set_description("{} MB".format(2 * size_used // (1024 * 1024)))
                        print("Chunk Saved!--------------------------------------------------------")

        if(current_chunk_src is not None):
            np.save(os.path.join(src_output_path, "{}.npy".format(index)), current_chunk_src)
            np.save(os.path.join(dst_output_path, "{}.npy".format(index)), current_chunk_dst)
            np.save(os.path.join(mask_output_path, "{}.npy".format(index)), current_chunk_mask)

            for i in range(prev_range_end + 1, cur_range_end + 1):
                indices[i] = (index, prev_range_end + 1)
                    
            savePickle(indices_path, indices)
            prev_range_end = cur_range_end
            current_chunk_src = None
            current_chunk_dst = None
            current_chunk_mask = None
            index += 1
            size_used += cur_chunk_size
            prog_bar.set_description("{} MB".format(2 * size_used // (1024 * 1024)))
            print("Chunk Saved!--------------------------------------------------------")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)

    print("Done! with total size used {} MB".format(2 * size_used // (1024 * 1024)))