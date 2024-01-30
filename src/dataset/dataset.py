from pathlib import Path
import h5py
import torch
import numpy as np
import torchvision
from src.dataset.augmentations import RandomRotTranslTransform, SimplexNoiseTransform, RandomHorizontalFlip, ColorJitterAugmentation, GrayScaleAugmentation


class H5Dataset(torch.utils.data.Dataset):
    LED_TYPES = ["bb", "bl", "br", "bf", "tl", "tr"]
    LED_VISIBILITY_RANGES_DEG = [
        [[-180, -100], [100, 180]],
        [[35, 145], [np.inf, np.inf]],
        [[-145, -35], [np.inf, np.inf]],
        [[-80, 80], [np.inf, np.inf]],
        [[0, 180], [np.inf, np.inf]],
        [[-180, -0], [np.inf, np.inf]],
    ]
    POS_ORB_SIZE = 20

    # Use this for good visibility
    # LED_VISIBILITY_RANGES_DEG = [
    #     [[-180, -130], [130, 180]],
    #     [[75, 95], [np.inf, np.inf]],
    #     [[-95, -75], [np.inf, np.inf]],
    #     [[-50, 50], [np.inf, np.inf]],
    #     [[30, 150], [np.inf, np.inf]],
    #     [[-150, -30], [np.inf, np.inf]],
    # ]
    LED_VISIBILITY_RANGES_RAD = np.deg2rad(LED_VISIBILITY_RANGES_DEG)

    def __init__(self, filename, keys=None,
                 transform=lambda x: x, libver='latest', target_robots = None,
                 only_visible_robots = False,
                 robot_id = None,
                 sample_count = None,
                 sample_count_seed = None,
                 compute_led_visibility = False,
                 supervised_flagging = None,
                 supervised_flagging_seed = None,
                 distance_range = None):
        
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(
                f'Dataset "{filename}" does not exist')

        self.h5f = h5py.File(filename, 'r', libver=libver)
        self.keys = keys
        self.data = self.h5f
        self.transform = transform
        self.robot_ids = set(self.data.attrs["RIDs"])
        self.robot_ids_int = [int(rid[2:]) for rid in self.robot_ids]
        self.compute_visibility_mask = compute_led_visibility

        self.proj_uvz_keys = {}
        self.pose_rel_keys = {}
        self.led_keys = {}

        self.valid_ds_indexes = torch.arange(self.data["robot_id"].shape[0])

        if target_robots is None:
            target_robots = self.robot_ids_int
        else:
            target_robots = [int(tr[2:]) for tr in target_robots]


        for source_rid in self.robot_ids_int:
            self.proj_uvz_keys[source_rid] = []
            self.pose_rel_keys[source_rid] = []
            for target_rid in self.robot_ids_int:
                if source_rid != target_rid:
                    if target_rid in target_robots:
                        col_name = f"RM{source_rid}_proj_uvz_RM{target_rid}"
                        pose_rel_col_name = f"RM{source_rid}_pose_rel_RM{target_rid}"
                        self.proj_uvz_keys[source_rid].append(col_name)
                        self.pose_rel_keys[source_rid].append(pose_rel_col_name)
                        for led_key in self.LED_TYPES:
                            if not self.led_keys.get(source_rid):
                                self.led_keys[source_rid] = []
                            self.led_keys[source_rid].append(f"RM{int(target_rid)}_led_{led_key}")
        

        # Are we filtering out for visible robots only?
        
        self.visibility_mask = torch.zeros(len(self), dtype=torch.int8)

        bounds = np.array([
            [0, 640], # u
            [0, 360], # v
            [0, np.inf] # z
        ])
        for i in range(len(self)):
            keys = self.proj_uvz_keys[self.data["robot_id"][i]]
            for k in keys:
                # breakpoint()
                if (self.data[k][i] > bounds[:, 0]).all() and (self.data[k][i] < bounds[:, 1]).all():
                    self.visibility_mask[i] = True
                    break
            
        distance_mask = np.ones(len(self), dtype=np.int8)

        if distance_range:
            print("DISTANCE RANGE FILTERING IS DISABLED")
        #     distance_true = self.data["pose_rel"][:, 2]
        #     distance_mask = (distance_true <= distance_range[1]) &\
        #     (distance_true >= distance_range[0])
        
        
        if only_visible_robots:
            self.valid_ds_indexes = np.where(self.visibility_mask & distance_mask)[0]
        else:
            self.valid_ds_indexes = np.where(distance_mask)[0]

        if robot_id:
            mask = self.data["robot_id"][self.valid_ds_indexes] == robot_id
            self.valid_ds_indexes = self.valid_ds_indexes[mask]
    
            
        if sample_count:
            sample_count_seed = sample_count_seed if sample_count_seed else 0
            if sample_count_seed < 0:
                self.valid_ds_indexes = self.valid_ds_indexes[:sample_count]
            else:
                np.random.seed(sample_count_seed)
                self.valid_ds_indexes = np.random.choice(self.valid_ds_indexes,
                                                        size=sample_count,
                                                        replace=False)

        self.__pos_map_orb = self.__pos_orb(self.POS_ORB_SIZE)


        self.supervised_mask = torch.ones(self.data["robot_id"].shape, dtype=torch.bool)

        if supervised_flagging is not None:
            np.random.seed(supervised_flagging_seed)
            supervised_indexes = np.random.choice(np.where(self.visibility_mask)[0],
                                                       size = supervised_flagging,
                                                       replace=False)
            self.supervised_mask = np.zeros_like(self.data["robot_id"], dtype=bool)
            self.supervised_mask[supervised_indexes.tolist()] = True
            

    def __getitem__(self, slice):
        slice = self.valid_ds_indexes[slice]
        # This is tailored for 2 total robots. Subject to change in the future
        slice_robot_id = self.data["robot_id"][slice]

        batch = {}
        # batch['proj_x'] = np.array(self.data[f"RM{slice_robot_id}_proj_x"][slice])
        
        for proj_uvz_key in self.proj_uvz_keys[slice_robot_id]:
            batch["proj_uvz"] = torch.tensor(self.data[proj_uvz_key][slice])
        
        for pose_rel_key in self.pose_rel_keys[slice_robot_id]:
            batch["pose_rel"] = torch.tensor(self.data[pose_rel_key][slice])
    
        batch["led_mask"] = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int8)
        for i, led_key in enumerate(self.led_keys[slice_robot_id]):
            batch[led_key[4:]] = torch.tensor([self.data[led_key][slice]], dtype=torch.uint8)
            batch["led_mask"][i] = torch.tensor([self.data[led_key][slice]], dtype=torch.uint8)

        for robot_id in self.robot_ids:
            batch[robot_id + "_pose"] = torch.tensor(self.data[robot_id + "_pose"][slice].squeeze())

        
        batch['image'] = torch.tensor((self.data["image"][slice].astype(np.float32) / 255.).transpose(2, 0, 1))
        
        batch["timestamp"] = torch.tensor([self.data["timestamp"][slice]])
        # u_visible = (batch['proj_uvz'][0] > 0 - self.POS_ORB_SIZE // 2) & (batch['proj_uvz'][0] < 640 + self.POS_ORB_SIZE // 2)
        # v_visible = (batch['proj_uvz'][1] > 0 - self.POS_ORB_SIZE // 2) & (batch['proj_uvz'][1] < 360 + self.POS_ORB_SIZE // 2)
        # z_visible = (batch['proj_uvz'][2] > 0)
        batch['robot_visible'] = self.visibility_mask[slice]
        batch['pos_map'] = torch.tensor(self.__position_map(batch["proj_uvz"], batch['robot_visible'], orb_size=self.POS_ORB_SIZE))
        # batch["distance_rel"] = torch.linalg.norm(batch["pose_rel"][:-1]).squeeze()
        batch["distance_rel"] = batch["pose_rel"][0]
        batch["robot_id"] = torch.tensor([slice_robot_id])

        batch["supervised_flag"] = self.supervised_mask[slice]

        if self.compute_visibility_mask:
            other_pose_rel = self.data[self.__other_rid('RM' + str(slice_robot_id)) + "_pose_rel_RM" + str(slice_robot_id)][slice]
            other_theta_rel = np.arctan2(other_pose_rel[1], other_pose_rel[0])
            led_visibility = (other_theta_rel >= self.LED_VISIBILITY_RANGES_RAD[:, :, 0]) &\
                (other_theta_rel <= self.LED_VISIBILITY_RANGES_RAD[:, :, 1])
            batch["led_visibility_mask"] = torch.tensor((led_visibility[:, 0] | led_visibility[:, 1]).tolist())
    
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
        # return 1 - np.tanh(.04 * distances)
        return distances <= orb_size / 2


    def __position_map(self, proj_uvz, robot_visible, map_size = (360, 640), orb_size = 20, align_to = None):
        # Padded so i can easily crop it out
        if not robot_visible:
            return np.ones(map_size)

        padding = orb_size
        result = np.pad(np.zeros(map_size), padding, 'constant', constant_values=0) # 440x720
        
        if proj_uvz[-1] > 0:
            u = 0
            v = 0

            if not align_to:
                u = int(proj_uvz[0]) + padding
                v = int(proj_uvz[1]) + padding
            else:
                u = int(proj_uvz[0]) // align_to * align_to + align_to // 2 + padding
                v = int(proj_uvz[1]) // align_to * align_to + align_to // 2 + padding
            result[v - orb_size // 2 : v + orb_size // 2, u - orb_size // 2 : u + orb_size // 2] = self.__pos_map_orb
        return result[padding:-padding, padding:-padding]
    
    def __other_rid(self, rid):
        return list(self.robot_ids - set([rid,]))[0]
    

def get_dataset(dataset_path, camera_robot = None, target_robots = None, augmentations = False,
                sample_count = None, sample_count_seed = None,
                only_visible_robots = False,
                compute_led_visibility = False,
                supervised_flagging = None,
                supervised_flagging_seed = None,
                distance_range = None):
    
    transform = lambda x: x
    if augmentations:
        transform = torchvision.transforms.Compose([
            # RandomHorizontalFlip((360, 640)),
            RandomRotTranslTransform(9, .1, bound=H5Dataset.POS_ORB_SIZE * 2),
            SimplexNoiseTransform((360, 640)),
            ColorJitterAugmentation(
               brightness=.4,
               hue=.2
           )
        ])

    camera_robot_id_int = None
    if camera_robot:
        camera_robot_id_int = int(camera_robot[2:])

    dataset = H5Dataset(dataset_path, target_robots = target_robots,
                        transform=transform, only_visible_robots = only_visible_robots,
                        robot_id=camera_robot_id_int,
                        sample_count = sample_count,
                        sample_count_seed = sample_count_seed,
                        compute_led_visibility = compute_led_visibility,
                        supervised_flagging=supervised_flagging,
                        supervised_flagging_seed=supervised_flagging_seed,
                        distance_range = distance_range)

    mask = torch.ones(len(dataset), dtype=torch.bool)
    

    return torch.utils.data.Subset(dataset, torch.arange(len(dataset))[mask])



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from time import time

    dataset = H5Dataset("../robomaster_led/robomaster_ds_full_on_training.h5")
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    counts = {}
    start_time = time()
    visible_robots = 0
    for batch in iter(dataloader):
        for k in batch.keys():
            counts[k] = counts.get(k, 0) + 1
        visible_robots += batch['robot_visible'][0].cpu().int()
    elapsed = time() - start_time
    print(f"Read whole dataset in {elapsed:.2f} seconds")
    print(counts)
    print(f"Visible instances: {visible_robots}")
