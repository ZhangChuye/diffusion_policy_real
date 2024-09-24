import h5py

def get_hdf5_metadata(file_path):
    """
    Recursively retrieves the metadata of the HDF5 file, including datasets, shapes, and types.
    
    Args:
        file_path (str): Path to the HDF5 file.
        
    Returns:
        metadata (dict): Dictionary containing the metadata of the file.
    """
    metadata = {}

    def extract_metadata(name, obj):
        if isinstance(obj, h5py.Dataset):
            metadata[name] = {
                "shape": obj.shape,
                "dtype": str(obj.dtype)
            }

    with h5py.File(file_path, 'r') as f:
        f.visititems(extract_metadata)
    
    return metadata

def compare_hdf5_metadata(file1_meta, file2_meta):
    """
    Compares the metadata of two HDF5 files.
    
    Args:
        file1_meta (dict): Metadata of the first HDF5 file.
        file2_meta (dict): Metadata of the second HDF5 file.
    """
    all_keys = set(file1_meta.keys()).union(file2_meta.keys())

    for key in all_keys:
        meta1 = file1_meta.get(key)
        meta2 = file2_meta.get(key)

        if key not in file1_meta:
            print(f"'{key}' exists only in the second file.")
        elif key not in file2_meta:
            print(f"'{key}' exists only in the first file.")
        else:
            if meta1["shape"] != meta2["shape"]:
                print(f"Shape mismatch for '{key}': {meta1['shape']} vs {meta2['shape']}")
            if meta1["dtype"] != meta2["dtype"]:
                print(f"Dtype mismatch for '{key}': {meta1['dtype']} vs {meta2['dtype']}")
            if meta1["shape"] == meta2["shape"] and meta1["dtype"] == meta2["dtype"]:
                print(f"'{key}' matches in shape and dtype.")


if __name__ == "__main__":
    file1_path = "/mnt/home/ZhangChuye/ATM_Data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/bc_train_10/demo_0.hdf5"
    file2_path = "/mnt/home/ZhangChuye/Data/dp_pick_up_cube/224_224_3/episode_0.hdf5"

    # Get metadata of both files
    file1_metadata = get_hdf5_metadata(file1_path)
    file2_metadata = get_hdf5_metadata(file2_path)

    # print metadata
    print("@"*10)
    
    print(file1_metadata)
    print("@"*10)
    
    print(file2_metadata)
    print("@"*10)

    # Compare metadata
    # compare_hdf5_metadata(file1_metadata, file2_metadata)