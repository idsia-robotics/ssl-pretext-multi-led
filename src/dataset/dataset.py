from pathlib import Path
import h5py
import torch
import numpy as np
import torchvision
from src.dataset.augmentations import RandomRotTranslTransform, SimplexNoiseTransform, RandomHorizontalFlip


class PositionMapComputing:

    def __init__(self, orb_size) -> None:
        self.orb_size = orb_size
        self.pos_orb = self.__pos_orb(self.orb_size)

    def __pos_orb(self, orb_size = 20):
        U, V = np.meshgrid(
            np.arange(orb_size),
            np.arange(orb_size),
            indexing='xy'
        )
        grid = np.stack([V, U], axis = 0)
        grid -= orb_size // 2
        distances = np.linalg.norm(grid, axis = 0, ord = 2)
        return 1 - np.tanh(.1 * distances)
        
class H5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, filename, keys=None,
                 transform=lambda x: x, libver='latest', target_robots = None,
                 only_visible_robots = False,
                 robot_id = None,
                 sample_count = None,
                 sample_count_seed = None):
        
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(
                f'Dataset "{filename}" does not exist')

        self.h5f = h5py.File(filename, 'r', libver=libver)
        self.keys = keys
        self.data = self.h5f
        self.transform = transform
        self.robot_ids = self.data.attrs["RIDs"]
        self.robot_ids_int = [int(rid[2:]) for rid in self.robot_ids]

        self.proj_uvz_keys = {}
        self.led_keys = {}

        self.valid_ds_indexes = torch.arange(self.data["robot_id"].shape[0])

        if target_robots is None:
            target_robots = self.robot_ids_int
        else:
            target_robots = [int(tr[2:]) for tr in target_robots]


        for source_rid in self.robot_ids_int:
            self.proj_uvz_keys[source_rid] = []
            for target_rid in self.robot_ids_int:
                if source_rid != target_rid:
                    if target_rid in target_robots:
                        col_name = f"RM{source_rid}_proj_uvz_RM{target_rid}"
                        self.proj_uvz_keys[source_rid].append(col_name)
                        for led_key in ["bb", "bl", "br", "bf", "tl", "tr"]:
                            if not self.led_keys.get(source_rid):
                                self.led_keys[source_rid] = []
                            self.led_keys[source_rid].append(f"RM{int(target_rid)}_led_{led_key}")
        

        # Are we filtering out for visible robots only?
        
        if only_visible_robots:
            visibility_mask = np.zeros(len(self), dtype=np.int8)
            bounds = np.array([
                [0, 360],
                [0, 640],
                [0, np.inf]
            ])
            for i in range(len(self)):
                keys = self.proj_uvz_keys[self.data["robot_id"][i]]
                for k in keys:
                    # breakpoint()
                    if (self.data[k][i] > bounds[:, 0]).all() and (self.data[k][i] < bounds[:, 1]).all():
                        visibility_mask[i] = True
                        break
            self.valid_ds_indexes = np.where(visibility_mask > 0)[0]

        if robot_id:
            mask = self.data["robot_id"][self.valid_ds_indexes] == robot_id
            self.valid_ds_indexes = self.valid_ds_indexes[mask]
    
            
        if sample_count:
            print("Selecting subset")
            np.random.seed(sample_count_seed)
            self.valid_ds_indexes = np.random.choice(self.valid_ds_indexes,
                                                     size=sample_count,
                                                     replace=False)

        self.POS_ORB_SIZE = 40
        self.__pos_map_orb = self.__pos_orb(self.POS_ORB_SIZE)

    def __getitem__(self, slice):
        slice = self.valid_ds_indexes[slice]
        # This is tailored for 2 total robots. Subject to change in the future
        slice_robot_id = self.data["robot_id"][slice]
        batch = {}
        
        for proj_key in self.proj_uvz_keys[slice_robot_id]:
            batch["proj_uvz"] = torch.tensor(self.data[proj_key][slice])
        
        for led_key in self.led_keys[slice_robot_id]:
            batch[led_key[4:]] = int(self.data[led_key][slice])
        
        batch['image'] = torch.tensor((self.data["image"][slice].astype(np.float32) / 255.).transpose(2, 0, 1))
        
        batch["timestamp"] = self.data["timestamp"][slice]
        u_visible = (batch['proj_uvz'][0] > 0 - self.POS_ORB_SIZE // 2) & (batch['proj_uvz'][0] < 640 + self.POS_ORB_SIZE // 2)
        v_visible = (batch['proj_uvz'][1] > 0 - self.POS_ORB_SIZE // 2) & (batch['proj_uvz'][1] < 360 + self.POS_ORB_SIZE // 2)
        z_visible = (batch['proj_uvz'][2] > 0)
        batch['robot_visible'] = (u_visible & v_visible & z_visible)
        batch['pos_map'] = torch.tensor(self.__position_map(batch["proj_uvz"], batch['robot_visible'], orb_size=self.POS_ORB_SIZE))
        return self.transform(batch)
    
    def __len__(self):
        return self.valid_ds_indexes.shape[0]
    
    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()

    def __pos_orb(self, orb_size = 20):
        U, V = np.meshgrid(
            np.arange(orb_size),
            np.arange(orb_size),
            indexing='xy'
        )
        grid = np.stack([V, U], axis = 0)
        grid -= orb_size // 2
        distances = np.linalg.norm(grid, axis = 0, ord = 2)
        return 1 - np.tanh(.1 * distances)


    def __position_map(self, proj_uvz, robot_visible, map_size = (360, 640), orb_size = 20):
        # Padded so i can easily crop it out
        if not robot_visible:
            return np.zeros(map_size)

        padding = orb_size
        result = np.pad(np.zeros(map_size), padding, 'constant', constant_values=0) # 440x720
        
        if proj_uvz[-1] > 0:
            u = int(proj_uvz[0]) + padding
            v = int(proj_uvz[1]) + padding
            try:
                result[v - orb_size // 2 : v + orb_size // 2, u - orb_size // 2 : u + orb_size // 2] = self.__pos_map_orb
            except:
                breakpoint()
        return result[padding:-padding, padding:-padding]
    

def get_dataset(dataset_path, camera_robot = None, target_robots = None, augmentations = False,
                sample_count = None, sample_count_seed = None,
                only_visible_robots = False):
    
    transform = lambda x: x
    if augmentations:
        transform = torchvision.transforms.Compose([
            RandomHorizontalFlip((360, 640)),
            RandomRotTranslTransform(9, .1),
            SimplexNoiseTransform((360, 640))
        ])

    camera_robot_id_int = None
    if camera_robot:
        camera_robot_id_int = int(camera_robot[2:])

    dataset = H5Dataset(dataset_path, target_robots = target_robots,
                        transform=transform, only_visible_robots = only_visible_robots,
                        robot_id=camera_robot_id_int,
                        sample_count = sample_count,
                        sample_count_seed = sample_count_seed)

    mask = torch.ones(len(dataset), dtype=torch.bool)
    # if sample_count:
    #     current_picked_idx = torch.where(mask)[0]
    #     np.random.seed(sample_count_seed if sample_count_seed else 0)
    #     current_picked_idx = np.random.choice(current_picked_idx, sample_count,
    #                      replace=False)
    #     mask[:] = False
    #     mask[current_picked_idx] = True

    

    return torch.utils.data.Subset(dataset, torch.arange(len(dataset))[mask])



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from time import time

    dataset = H5Dataset("data/robomaster_ds_train.h5")
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    counts = {}
    start_time = time()
    visible_robots = 0
    for batch in iter(dataloader):
        for k in batch.keys():
            counts[k] = counts.get(k, 0) + 1
        visible_robots += batch['robot_visible']
    elapsed = time() - start_time
    print(f"Read whole dataset in {elapsed:.2f} seconds")
    print(counts)
    print("Visible instances: {visible_robots}")
